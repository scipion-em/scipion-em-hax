# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreos@cnb.csic.es) [1]
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

class JaxProtFilterLatents(ProtAnalysis3D, ProtFlexBase):
    """ Protocol to filter automatically latent conformational spaces"""
    _label = 'filter latent space'
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
        group.addParam('inputParticles', params.PointerParam, label="Input particles",
                       pointerClass='SetOfParticlesFlex', important=True)

        group.addParam('batchSize', params.IntParam, default=1024, label='Number of images in batch',
                       help="Determines how many images will be load in the GPU at any moment during training (set by "
                            "default to 1024 - you can control GPU memory usage easily by tuning this parameter to fit your "
                            "hardware requirements - we recommend using tools like nvidia-smi to monitor and/or measure "
                            "memory usage and adjust this value - keep also in mind that bigger batch sizes might be "
                            "less precise when looking for very local motions")

        group = form.addGroup("Filter parameters")
        group.addParam("neighbours", params.IntParam, default=10, label="Number of neighbours",
                      help="Number of nearest neighbours to be search for each landscape sample. Smaller "
                           "values will better approximate the locality features of a sample, while global "
                           "values will capture more general landscape features.")

        group.addParam('outliersThreshold', params.FloatParam, default=1.0,
                      label="Outliers distance threshold",
                      help='Z-Score value from 0 to infinite. Only coordinates with a Z-Score smaller than '
                           'or equal to the threshold will be kept in the output')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.writeLatentsStep)
        self._insertFunctionStep(self.filterLatentsStep)
        self._insertFunctionStep(self.createOutputStep)

    def writeLatentsStep(self):
        inputParticles = self.inputParticles.get()

        # Save latents to .npy file
        latents = []
        for particle in inputParticles.iterItems():
            latents.append(particle.getZFlex())
        latents = np.vstack(latents)
        latents_file = self._getExtraPath("latents.npy")
        np.save(latents_file, latents)


    def filterLatentsStep(self):
        latents_file = self._getExtraPath("latents.npy")
        thr = self.outliersThreshold.get()
        n_neighbours = self.neighbours.get()
        batch_size = self.batchSize.get()


        args = (f"--latents {latents_file} --thr {thr} --n_neighbours {n_neighbours} --return_ids "
                f"--batch_size {batch_size} --output_path {self._getExtraPath()}")

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        program = hax.Plugin.getProgram("filter_latents", gpu)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        inputSet = self.inputParticles.get()

        filter_ids = np.load(self._getExtraPath("filtered_latents.npy")).astype(int)
        particle_ids = list(inputSet.getIdSet())

        partSet = self._createSetOfParticlesFlex(progName=inputSet.getFlexInfo().getProgName())
        partSet.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())

        for idx in filter_ids:
            partSet.append(inputSet[particle_ids[idx]].clone())

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(inputSet, partSet)


    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        return summary

    # ----------------------- VALIDATE functions -----------------------
    def _validate(self):
        """ Try to find errors on define params. """
        errors = []
        return errors

    def _warnings(self):
        warnings = []
        return warnings
