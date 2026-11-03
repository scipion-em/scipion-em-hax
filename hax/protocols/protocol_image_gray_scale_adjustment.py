# **************************************************************************
# *
# * Authors:     Eduardo García Delgado (eduardo.garcia@cnb.csic.es) [1]
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

from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.object import Float
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_1
from pyworkflow.utils import getExt, makePath

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import Volume, Particle

import xmipp3
from xmipp3.convert import writeSetOfParticles, matrixFromGeometry

import hax

class JaxProtImageAdjustment(ProtAnalysis3D, ProtFlexBase):
    """ Protocol to adjust the gray scale of each particle to the projections of a reference volume
    (closed form fit, no network)."""
    _label = 'adjust gray scale - images'
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

        group = form.addGroup("Data")
        group.addParam('inputParticles', params.PointerParam, label="Input particles", pointerClass='SetOfParticles',
                       important=True)

        group.addParam('inputVolume', params.PointerParam,
                       label="Starting volume", pointerClass='Volume',
                       help="Reference volume whose projections set the gray scale of each particle.")

        group.addParam('inputVolumeMask', params.PointerParam,
                       label="Reconstruction mask", pointerClass='VolumeMask', allowsNull=True,
                       help="Optional mask: the fit only uses the pixels inside its projection. If not provided, the "
                            "whole image is used.")

        group.addParam('ctfType', params.EnumParam, choices=['None', 'Apply', 'Wiener', 'Precorrect'],
                       default=1, label="CTF correction type",
                       display=params.EnumParam.DISPLAY_HLIST,
                       expertLevel=params.LEVEL_ADVANCED,
                       help="* *None*: the particles carry no CTF\n"
                            "* *Apply*, *Wiener*, *Precorrect*: the particles carry their CTF, which is applied to the "
                            "projections of the reference volume")

        form.addSection(label='Adjustment')
        form.addParam('contrastMode', params.EnumParam, choices=['relative', 'absolute'], default=0,
                      display=params.EnumParam.DISPLAY_HLIST,
                      label='Per image contrast',
                      help="* *relative*: each particle is divided by its gray scale relative to the mean of the set, so "
                           "the contrast becomes even across particles while the overall gray level is kept\n"
                           "* *absolute*: each particle is divided by its own gray scale, which puts every particle on "
                           "the gray scale of the reference volume")

        form.addParam('fitOffset', params.BooleanParam, default=False,
                      label='Fit an offset per image?',
                      help="Also fit and subtract an offset per particle (image = scale * projection + offset). If set "
                           "to No, only the scale is fitted, which assumes particles normalized to a zero-mean background.")

        form.addParam('scratchFolder', params.PathParam,
                      label='Path to SSD scratch folder',
                      help='Optional folder in a SSD/NVME disk where the particles are cached and read from. Recommended '
                           'when the particles are stored in a slow (HDD) disk.')

        form.addParam('batchSize', params.IntParam, default=256, label='Number of images in batch',
                      help="How many particles are projected at once. Lower it if the GPU runs out of memory; it does "
                           "not change the result.")

        form.addParallelSection(threads=0, mpi=4)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'imgsFn': self._getExtraPath('input_particles.xmd'),
            'predictedFn': self._getExtraPath('predicted_latents.xmd'),
            'fnVol': self._getExtraPath('volume.mrc'),
            'fnVolMask': self._getExtraPath('mask.mrc'),
        }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.writeMetaDataStep)
        self._insertFunctionStep(self.adjustStep)
        self._insertFunctionStep(self.createOutputStep)

    def writeMetaDataStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnVolMask = self._getFileName('fnVolMask')

        inputParticles = self.inputParticles.get()
        Xdim = inputParticles.getXDim()

        ih = ImageHandler()
        inputVolume = self.inputVolume.get().getFileName()
        ih.convert(self._getXmippFileName(inputVolume), fnVol)
        curr_vol_dim = ImageHandler(self._getXmippFileName(inputVolume)).getDimensions()[-1]
        if curr_vol_dim != Xdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --fourier %d " % (fnVol, Xdim), numberOfMpi=1,
                        env=xmipp3.Plugin.getEnviron())
        ih.setSamplingRate(fnVol, inputParticles.getSamplingRate())

        if self.inputVolumeMask.get():
            ih = ImageHandler()
            inputMask = self.inputVolumeMask.get().getFileName()
            if inputMask:
                ih.convert(self._getXmippFileName(inputMask), fnVolMask)
                curr_mask_dim = ImageHandler(self._getXmippFileName(inputMask)).getDimensions()[-1]
                if curr_mask_dim != Xdim:
                    self.runJob("xmipp_image_resize",
                                "-i %s --dim %d --interp nearest" % (fnVolMask, Xdim), numberOfMpi=1,
                                env=xmipp3.Plugin.getEnviron())

        writeSetOfParticles(inputParticles, imgsFn)

        # Write extra attributes (if needed)
        md = XmippMetaData(imgsFn)
        if hasattr(inputParticles.getFirstItem(), "_xmipp_subtomo_labels"):
            labels = np.asarray([int(particle._xmipp_subtomo_labels) for particle in inputParticles.iterItems()])
            md[:, "subtomo_labels"] = labels
        md.write(imgsFn, overwrite=True)

    def adjustStep(self):
        md_file = self._getFileName('imgsFn')
        vol_file = self._getFileName('fnVol')
        mask_file = self._getFileName('fnVolMask')
        sr = self.inputParticles.get().getSamplingRate()
        args = "--md %s --vol %s --sr %f --batch_size %d --per_image_contrast %s --output_path %s " \
               % (md_file, vol_file, sr, self.batchSize.get(), self.getEnumText('contrastMode'), self._getExtraPath())

        if self.inputVolumeMask.get():
            args += '--mask %s ' % mask_file

        if self.ctfType != 0:
            if self.ctfType.get() == 1:
                args += '--ctf_type apply '
            elif self.ctfType.get() == 2:
                args += '--ctf_type wiener '
            elif self.ctfType.get() == 3:
                args += '--ctf_type precorrect '
        else:
            args += '--ctf_type None '

        if self.fitOffset:
            args += '--fit_offset '

        if self.scratchFolder.get() is not None:
            args += '--ssd_scratch_folder %s ' % self.scratchFolder.get()

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        program = hax.Plugin.getProgram("image_gray_scale_adjustment", gpu)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()
        out_md = os.path.join(self._getExtraPath(), "adjusted_images.xmd")

        md = XmippMetaData(out_md)

        partSet.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())
        partSet.setAlignmentProj()

        idx = 0
        for particle in inputSet.iterItems():
            outParticle = Particle()
            outParticle.copyInfo(particle)

            index, fileName = md[idx, "image"].split("@")
            outParticle.setLocation(int(index), fileName)
            outParticle.a_adjustment = Float(md[idx, "adjustment_a"])
            outParticle.b_adjustment = Float(md[idx, "adjustment_b"])

            partSet.append(outParticle)
            idx += 1

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(inputSet, partSet)

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

        vol = self.inputVolume.get()
        mask = self.inputVolumeMask.get()

        if vol is None:
            errors.append("A volume is required. Please, select a valid and adequate volume for your dataset")

        if mask is not None:
            data = ImageHandler(mask.getFileName()).getData()
            if not np.all(np.logical_and(data >= 0, data <= 1)):
                errors.append("Mask provided is not binary. Please, provide a binary mask")

        return errors

    def _warnings(self):
        warnings = []

        return warnings

    # --------------------------- UTILS functions -----------------------

    def _getXmippFileName(self, filename):
        if getExt(filename) == ".mrc":
            filename += ":mrc"
        return filename