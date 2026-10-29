# **************************************************************************
# *
# * Authors:     Eduardo Garcia Delgado (eduardo.garcia@cnb.csic.es) [1]
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

from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow import VERSION_1
import pyworkflow.utils as pwutils

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import Volume, AtomStruct
import hax


class JaxProtRigidFitting(ProtAnalysis3D, ProtFlexBase):
    """ Protocol for rigid fitting models towards 3D Cryo-EM Maps."""
    _label = 'rigid fitting'
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

        form.addParam('inputStructure', params.PointerParam, important=True,
                       label="Atomic structure", pointerClass=AtomStruct,
                       help="Atomic structure imported as pdb or cif.")

        form.addParam('inputVolume', params.PointerParam, important=True,
                       label="Input volume", pointerClass='Volume',
                       help="Input volume for rigid fitting with atomic structure")

        form.addParam('isAligned', params.BooleanParam, default=False, label='Is already aligned?',
                      help='When set to Yes, it will be taken into account the previous alignment and, therefore, '
                           'only this method will serve as a validation for rigid fitted atomic structures into pre-aligned systems.')

        form.addParam('topkNMS', params.IntParam, default=10,
                      label='Maximum number of highest CC matches',
                      help='The higher the number, the most probable match might appear at the expense of computational time.')

        form.addParallelSection(threads=1, mpi=4)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'fnVol': self._getExtra('volume.mrc')
        }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.writeMetaDataStep)
        self._insertFunctionStep(self.rigidFittingStep)
        self._insertFunctionStep(self.createOutputStep)

    def writeMetaDataStep(self):
        ih = ImageHandler()

        fnVol = self._getFileName('fnVol')
        inputVolume = self._getInputVolume()
        ih.convert(self._getXmippFileName(inputVolume.getFileName()), fnVol)
        ih.setSamplingRate(fnVol, inputVolume.getSamplingRate())

    def rigidFittingStep(self):
        pdb_file = self.inputStructure.get().getFileName()
        vol_file = self._getFileName('fnVol')
        out_path = self._getExtraPath('pdb_fitted')
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        sr = self.inputVolume.get().getSamplingRate()
        topk_nms = self.topkNMS.get()
        args = "--pdb %s --vol %s --sr %f --topk_nms %d --out_dir %s " \
               % (pdb_file, vol_file, sr, topk_nms, out_path)

        if self.isAligned.get():
            args += '--is_aligned '

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = ''

        program = hax.Plugin.getProgram("rigid_fitting", gpu)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        pdb_file = self.inputStructure.get().getFileName()
        _, ext = os.path.splitext(pdb_file)
        out_path_vol = os.path.join(self._getExtraPath('pdb_fitted'), 'final_rigid_input_norm.mrc')
        if not self.isAligned:
            outputPdb1 = AtomStruct()
            out_path_pdb1 = os.path.join(self._getExtraPath('pdb_fitted'), f'final_rigid_fitted{ext.lower()}')
            outputPdb1.setFileName(out_path_pdb1)
            self._defineOutputs(outputStructure=outputPdb1)

            outputPdb2 = AtomStruct()
            out_path_pdb2 = os.path.join(self._getExtraPath('pdb_fitted'), f'final_rigid_refined_fitted{ext.lower()}')
            outputPdb2.setFileName(out_path_pdb2)
            self._defineOutputs(outputStructureRefined=outputPdb2)

        else:
            outputPdb = AtomStruct()
            out_path_pdb = os.path.join(self._getExtraPath('pdb_fitted'), f'final_rigid_fitted{ext.lower()}')
            outputPdb.setFileName(out_path_pdb)
            self._defineOutputs(outputStructure=outputPdb)

        outVol = Volume()
        outVol.setSamplingRate(self.inputVolume.get().getSamplingRate())
        outVol.setFileName(out_path_vol)
        self._defineOutputs(outputVolume=outVol)

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

        return errors

    def _warnings(self):
        warnings = []

        return warnings

    # --------------------------- UTILS functions -----------------------

    def _getExtra(self, *paths):
        return os.path.abspath(self._getExtraPath(*paths))

    def _getInputVolume(self, mode=""):
        if mode == "half":
            return self.inputVolume.get().getHalfMaps().split(",")
        else:
            return self.inputVolume.get()

    def _getXmippFileName(self, filename):
        if pwutils.getExt(filename) == ".mrc":
            filename += ":mrc"
        return filename