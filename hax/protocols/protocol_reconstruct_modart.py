# **************************************************************************
# *
# * Authors:     David Herreros (dherreros@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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


import numpy as np
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params

from pwem.objects import Volume
from pwem.protocols import ProtReconstruct3D

from xmipp3.convert import writeSetOfParticles

import hax
import hax.constants as const


class JaxProtReconstructMoDART(ProtReconstruct3D):
    """    
    Reconstruct a volume using MoDART algorithm from a given SetOfParticles.
    """
    _label = 'reconstruct MoDART'

    def __init__(self, **args):
        ProtReconstruct3D.__init__(self, **args)

    #--------------------------- DEFINE param functions --------------------------------------------   
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addHidden(
            params.USE_GPU,
            params.BooleanParam,
            default=True,
            label="Use GPU for execution",
            help="This protocol has both CPU and GPU implementation.\
                                         Select the one you want to use.",
        )

        form.addHidden(
            params.GPU_LIST,
            params.StringParam,
            default="0",
            expertLevel=params.LEVEL_ADVANCED,
            label="Choose GPU IDs",
            help="Add a list of GPU devices that can be used",
        )

        group = form.addGroup("Data")
        group.addParam('inputParticles', params.PointerParam, pointerClass='SetOfParticles, SetOfParticlesFlex',
                      pointerCondition='hasAlignmentProj',
                      label="Input particles",  
                      help='Select the input images from the project.')

        group.addParam('initialMap', params.PointerParam, pointerClass='Volume',
                      label="Initial map",
                      allowsNull=True,
                      help='If provided, this map will be used as the initialization of the reconstruction '
                           'process. Otherwise, an empty volume will be used')

        group.addParam('recMask', params.PointerParam, pointerClass='VolumeMask',
                      allowsNull=True,
                      label="Reconstruction mask",
                      help="Mask used to restrict the reconstruction space to increase performance.")

        group.addParam('ctfType', params.EnumParam, choices=['None', 'Apply', 'Wiener', 'Precorrect'],
                       default=1, label="CTF correction type",
                       display=params.EnumParam.DISPLAY_HLIST,
                       expertLevel=params.LEVEL_ADVANCED,
                       help="* *None*: CTF will not be considered\n"
                            "* *Apply*: CTF is applied to the projection generated from the reference map\n"
                            "* *Wiener*: input particle is CTF corrected by a Wiener fiter\n"
                            "* *Precorrect: similar to Wiener but CTF has already been corrected")

        group = form.addGroup("Motion correction")
        group.addParam('doMotionCorrection', params.BooleanParam, default=False,
                      condition="inputParticles and isinstance(inputParticles, SetOfParticlesFlex)",
                      label="Correct motion blurred artifacts?",
                      help="Correct the conformation of the particles during the reconstruct process "
                           "to reduce motion blurred artifacts and increase resolution. Note that this "
                           "option requires that the particles come from HetSIREN (mass transport) or Zernike3Deep. "
                           "Otherwise, the parameter should be set to 'No'")

        group = form.addGroup("Symmetry")
        group.addParam('symmetryGroup', params.StringParam, default="c1",
                      label='Symmetry group',
                      help='If no symmetry is present, give c1')

        group = form.addGroup("Reconstruction modes")
        group.addParam('mode', params.EnumParam, choices=['Reconstruct', 'Gold standard'],
                      default=0, display=params.EnumParam.DISPLAY_HLIST,
                      label="Reconstruction mode",
                      help="\t * Reconstruct: usual reconstruction of a single volume using all the images "
                           "in the dataset\n"
                           "\t * Gold standard: volumes halves reconstruction for FSC and local resolution "
                           "computations\n")

        form.addSection(label='Data loading')
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

        group = form.addGroup("Network hyperparameters")
        group.addParam('batchSize', params.IntParam, default=64, label='Number of images in batch',
                       help="Determines how many images will be load in the GPU at any moment during training (set by "
                            "default to 64 - you can control GPU memory usage easily by tuning this parameter to fit your "
                            "hardware requirements - we recommend using tools like nvidia-smi to monitor and/or measure "
                            "memory usage and adjust this value - keep also in mind that bigger batch sizes might be "
                            "less precise when looking for very local motions")

        form.addParallelSection(threads=4, mpi=1)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'imgsFn': self._getExtraPath('input_particles.xmd'),
            'outputVol': self._getExtraPath('modart_map.mrc'),
            'outputHalfFirst': self._getExtraPath('modart_first_half.mrc'),
            'outputHalfSecond': self._getExtraPath('modart_second_half.mrc'),
        }
        self._updateFilenamesDict(myDict)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.writeMetaDataStep)
        self._insertFunctionStep(self.reconstructStep)
        self._insertFunctionStep(self.createOutputStep)


    #--------------------------- STEPS functions --------------------------------------------
    def writeMetaDataStep(self):
        imgsFn = self._getFileName('imgsFn')
        inputParticles = self.inputParticles.get()
        writeSetOfParticles(inputParticles, imgsFn)

    def reconstructStep(self):
        imgsFn = self._getFileName('imgsFn')
        sr = self.inputParticles.get().getSamplingRate()
        symmetryGroup = self.symmetryGroup.get()
        batchSize = self.batchSize.get()
        output_path = self._getExtraPath()

        args = (f"--md {imgsFn} --sr {sr} --symmetry_group {symmetryGroup} --batch_size {batchSize} "
                f"--output_path {output_path} ")

        if self.initialMap.get():
            args += f"--vol {self.initialMap.get().getFileName()} "

        if self.recMask.get():
            args += f"--mask {self.recMask.get().getFileName()} "

        if self.ctfType != 0:
            if self.ctfType.get() == 1:
                args += "--ctf_type apply "
            elif self.ctfType.get() == 2:
                args += "--ctf_type wiener "
            elif self.ctfType.get() == 3:
                args += "--ctf_type precorrect "
        else:
            args += "--ctf_type None "

        if self.mode.get() == 1:
            args += "--reconstruct_halves "

        if self.doMotionCorrection.get():
            args += f"--motion_correction {self.inputParticles.get().getFlexInfo().modelPath.get()} "

        if self.lazyLoad:
            args += "--load_images_to_ram "
        else:
            if self.scratchFolder.get() is not None:
                args += f"--ssd_scratch_folder {self.scratchFolder.get()} "

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = ''

        program = hax.Plugin.getProgram("modart", gpu)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        volume = Volume()
        volume.setFileName(self._getFileName('outputVol'))
        volume.setSamplingRate(self.inputParticles.get().getSamplingRate())

        if self.mode.get() == 1:
            volume.setHalfMaps([self._getFileName('modart_first_half'), self._getFileName('modart_second_half')])

        # Set correct sampling rate in volume header
        ImageHandler().setSamplingRate(self._getExtraPath("modart_map.mrc"), self.inputParticles.get().getSamplingRate())

        self._defineOutputs(outputVolume=volume)
        self._defineSourceRelation(self.inputParticles, volume)

    #--------------------------- INFO functions -------------------------------------------- 
    def _summary(self):
        """ Should be overriden in subclasses to 
        return summary message for NORMAL EXECUTION. 
        """
        return []

    def _validate(self):
        errors = []

        mask = self.recMask.get()
        if mask:
            data = ImageHandler(mask.getFileName()).getData()
            if not np.all(np.logical_or(data == 0, data == 1)):
                errors.append("Mask provided is not binary. Please, provide a binary mask")

        return errors

    #--------------------------- UTILS functions --------------------------------------------
    def _validate(self):
        """ Try to find errors on define params. """
        errors = []

        if self.doMotionCorrection.get():
            correction_method = self.inputParticles.get().getFlexInfo().getProgName()

            if correction_method not in [const.HETSIREN, const.ZERNIKE3D]:
                errors.append("Motion correction method not recognized. If you want to use motion correction, the particles "
                              "provided must come from HetSIREN (mass transport) or Zernike3Deep protocols.")

        return errors