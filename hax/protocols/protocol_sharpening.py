# **************************************************************************
# *
# * Authors:     Eduardo Garcia Delgado (eduardo.garcia@cnb.csic.es) [1]
# *              David Herreros Calero (dherreos@cnb.csic.es) [1]
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC [1]
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import os
import numpy as np
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_1
import pyworkflow.utils as pwutils

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import Volume

import xmipp3
import hax


class JaxProtGaussianSharpening(ProtAnalysis3D, ProtFlexBase):
    """ Protocol for sharpening with Gaussian Fitting reconstruction."""
    _label = 'gaussian sharpening'
    _lastUpdateVersion = VERSION_1

    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                                 Select the one you want to use.")

        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")

        form.addParam('inputVolume', params.PointerParam, important=True,
                       label="Starting volume", pointerClass='Volume',
                       help="Main input when applying for deconvolution, as it will provide all the necessary information to be "
                            "subtracted from. A reconstructed volume will come out of the difference between a dynamic and splatted "
                            "adjusted volume and this target volume, providing for the corresponding deconvolution.")

        form.addParam('inputVolumeMask', params.PointerParam,
                       label="Reconstruction mask", pointerClass='VolumeMask', important=True,
                       help="Reconstruction mask meant for encapsulate input volume when applying for deconvolution.")

        form.addParam('lazyLoad', params.BooleanParam, default=False,
                      expertLevel=params.LEVEL_ADVANCED, label='Lazy loading into RAM',
                      help='If provided, images will be loaded to RAM. This is recommended if you want the best performance '
                           'and your dataset fits in your RAM memory. If this flag is not provided, '
                           'images will be memory mapped. When this happens, the program will trade disk space for performance. '
                           'Thus, during the execution additional disk space will be used and the performance '
                           'will be slightly lower compared to loading the images to RAM. Disk usage will be back to normal '
                           'once the execution has finished.')

        form.addParam('scratchFolder', params.PathParam,
                      condition="not lazyLoad",
                      label='Path to SSD scratch folder',
                      help='If you are not loading the images to RAM, we strongly recommend to provide here a path to a folder in '
                           'a SSD/NVME disk to speed up the data loading. In general, you can expected a decrease in training performance when '
                           'loading images > 256px on a HDD disk.')

        form.addSection(label='Gaussian Fitting')
        form.addParam('gradWeight', params.FloatParam,
                       expertLevel=params.LEVEL_ADVANCED, label='Envelope smoothing', allowsNull=True,
                       help='This parameter can hold transitions between map and solvent accountable while training. The higher the value, the smoother '
                            'the structure will be. If not applied, it will start by default value 1.0.')

        form.addParam('lapWeight', params.FloatParam,
                       expertLevel=params.LEVEL_ADVANCED, label='High-Resolution detailing', allowsNull=True,
                       help='This parameter can hold high-resolution details accountable while training. The higher the value, the greater '
                            'the force it will apply on each voxel, enforcing an amplification of abrupt density changes but with the risk of '
                            'creating over-sharpened and unnatural structures within the volume. If not applied, it will start by default value 1.0.')

        form.addParam('tvWeight', params.FloatParam,
                       expertLevel=params.LEVEL_ADVANCED, label='Structural continuity', allowsNull=True,
                       help='This parameter can hold map continuity accountable while training. The higher the value, the more pressure it will apply on '
                            'the general densities of the structure, with the risk of over-performing an inverse process for high-resolution detailing and, '
                            'therefore, losing all possibles sharpening in the process. If not applied, it will start by default value 1.0.')

        form.addParam('sparsWeight', params.FloatParam,
                       label='Noise reductor', allowsNull=True,
                       help='This parameter can hold noise accountable while training. The higher the value, the more pressure it will apply on '
                            'the general noise of the structure, with the risk of losing true protein signal. If not applied, it will start '
                            'by default value 1.0.')

        form.addParam('boundWeight', params.FloatParam,
                       expertLevel=params.LEVEL_ADVANCED, label='Box boundary enforcement', allowsNull=True,
                       help='This parameter can hold voxel boundary accountable while training. It forces the model to strictly contain all density '
                            'within a specific boundary (normally, 90% of the whole box). If not applied, it will start by default value 1.0.')

        form.addParam('sigmaWeight', params.FloatParam,
                       expertLevel=params.LEVEL_ADVANCED, label='Nyquist limit enforcement', allowsNull=True,
                       help='This parameter can hold gaussians size accountable while training. The higher the value, the more pressure it will apply on '
                            'the global size of each gaussian, pushing them towards the specified limits (normally, between 20% and 200% of a pixel). '
                            'If not applied, it will start by default value 1.0.')

        form.addParam('l1Weight', params.FloatParam, default=0.2,
                       label='Outlier and sharpening weighter',
                       help='This parameter can hold outliers (noise) and smooth surfaces accountable while training. The higher the value, '
                            'the more force it will apply on structural outliers and it will prevent the network from oversmooth the structure, '
                            'with the risk of overfitting and creating unreal protein signal. Default value is a good starting point while '
                            'staying within an acceptable range [0.2, 0.8].')

        form.addParam('maxIterations', params.IntParam, default=20000,
                       label='Maximum number of iterations in fitting',
                       help='This specific number will give you the maximum length of the total gaussian fitting optimization. The greater the value, '
                            'the greater the convergence but accounting for the risk of overfitting. A proper number for good convergence would be '
                            'the one already in default, but it might depend on the structure to sharpen.')

        form.addParam('maxGaussians', params.IntParam, default=50000,
                       label='Maximum number of gaussians',
                       help='Limit number of gaussians to provide the network to work in the total gaussian fitting optimization. The greater the value, '
                            'the more freedom it is given to the network, taking into account that it may lead to overfitting.')

        form.addParallelSection(threads=1, mpi=4)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'fnVol': self._getExtra('volume.mrc'),
            'fnVolHalf1': self._getExtra('half_map_1.mrc'),
            'fnVolHalf2': self._getExtra('half_map_2.mrc'),
            'fnVolMask': self._getExtra('mask.mrc')
        }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.writeMetaDataStep)
        self._insertFunctionStep(self.trainingPredictStep)
        self._insertFunctionStep(self.createOutputStep)

    def writeMetaDataStep(self):
        fnVol = self._getFileName('fnVol')
        fnHalfMap1 = self._getFileName('fnVolHalf1')
        fnHalfMap2 = self._getFileName('fnVolHalf2')
        fnVolMask = self._getFileName('fnVolMask')

        ih = ImageHandler()

        inputVolume = self._getInputVolume()
        ih.convert(self._getXmippFileName(inputVolume.getFileName()), fnVol)
        ih.setSamplingRate(fnVol, inputVolume.getSamplingRate())

        halfMaps = self._getInputVolume(mode="half")
        ih.convert(self._getXmippFileName(halfMaps[0]), fnHalfMap1)
        ih.setSamplingRate(fnHalfMap1, inputVolume.getSamplingRate())
        ih.convert(self._getXmippFileName(halfMaps[1]), fnHalfMap2)
        ih.setSamplingRate(fnHalfMap2, inputVolume.getSamplingRate())

        inputMask = self.inputVolumeMask.get().getFileName()
        ih.convert(self._getXmippFileName(inputMask), fnVolMask)

    def trainingPredictStep(self):
        imgs_file = self._getExtra('input_particles')
        vol_file = self._getFileName('fnVol')
        half_map_1 = self._getFileName('fnVolHalf1')
        half_map_2 = self._getFileName('fnVolHalf2')
        mask_file = self._getFileName('fnVolMask')
        out_path = self._getExtra()
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        l1Weight = self.l1Weight.get()
        maxIterations = self.maxIterations.get()
        maxGaussians = self.maxGaussians.get()
        sr = self.inputVolume.get().getSamplingRate()
        args = ("--particles %s --vol %s --half_map_1 %s --half_map_2 %s --mask %s --sr %f --n_jobs %d "
                "--l1_weight %f --max_iterations %d --max_gaussians %d --output_path %s ") \
               % (imgs_file, vol_file, half_map_1, half_map_2, mask_file, sr, self.numberOfThreads.get(),
                  l1Weight, maxIterations, maxGaussians, out_path)

        if self.gradWeight.get() is not None:
            args += '--ratio_grad %f ' % self.gradWeight.get()

        if self.lapWeight.get() is not None:
            args += '--ratio_lap %f ' % self.lapWeight.get()

        if self.tvWeight.get() is not None:
            args += '--ratio_tv %f ' % self.tvWeight.get()

        if self.sparsWeight.get() is not None:
            args += '--ratio_spars %f ' % self.sparsWeight.get()

        if self.boundWeight.get() is not None:
            args += '--ratio_bound %f ' % self.boundWeight.get()

        if self.sigmaWeight.get() is not None:
            args += '--ratio_sigma %f ' % self.sigmaWeight.get()

        if self.lazyLoad:
            args += '--load_images_to_ram '
        else:
            if self.scratchFolder.get() is not None:
                args += '--ssd_scratch_folder %s ' % self.scratchFolder.get()

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = ''

        program = hax.Plugin.getProgram("gaussian_sharpening", gpu)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        inputVolume = self.inputVolume.get()
        outVol = Volume()
        outVol.setSamplingRate(inputVolume.getSamplingRate())
        ImageHandler().scaleSplines(os.path.join(self._getExtra(), f"consensus_volume.mrc"),
                                    os.path.join(self._getExtra(), f"consensus_volume.mrc"),
                                    finalDimension=inputVolume.getXDim(), overwrite=True)

        ImageHandler().setSamplingRate(os.path.join(self._getExtra(), f"consensus_volume.mrc"),
                                       inputVolume.getSamplingRate())

        outVol.setLocation(os.path.join(self._getExtra(), f"consensus_volume.mrc"))

        self._defineOutputs(outputVolumes=outVol)
        self._defineTransformRelation(inputVolume, outVol)

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        logFile = os.path.abspath(self._getLogsPath()) + "/run.stdout"
        with open(logFile, "r") as fi:
            for ln in fi:
                if ln.startswith("GPU memory has"):
                    summary.append(ln)
                    break
        return summary

    # ----------------------- VALIDATE functions -----------------------
    def _validate(self):
        """ Try to find errors on define params. """
        errors = []

        mask = self.inputVolumeMask.get()

        if mask is not None:
            data = ImageHandler(mask.getFileName()).getData()
            if not np.all(np.logical_and(data >= 0, data <= 1)):
                errors.append("Mask provided is not binary. Please, provide a binary mask")

        return errors

    def _warnings(self):
        warnings = []

        return warnings

    # --------------------------- UTILS functions -----------------------

    def _getExtra(self, *paths):
        return os.path.abspath(self._getExtraPath(*paths))

    def _getInputVolume(self, mode=""):
        if mode =="half":
            return self.inputVolume.get().getHalfMaps().split(",")
        else:
            return self.inputVolume.get()

    def _getXmippFileName(self, filename):
        if pwutils.getExt(filename) == ".mrc":
            filename += ":mrc"
        return filename