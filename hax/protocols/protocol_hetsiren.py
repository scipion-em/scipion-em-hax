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
from sklearn.cluster import KMeans

from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.object import String
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_1
from pyworkflow.utils import getExt, makePath

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import Volume, ParticleFlex
from pwem import ALIGN_PROJ

import xmipp3
from xmipp3.convert import writeSetOfParticles, matrixFromGeometry

import hax
import hax.constants as const

class JaxProtFlexibleAlignmentHetSiren(ProtAnalysis3D, ProtFlexBase):
    """ Protocol for angular alignment with heterogeneous reconstruction with the HetSIREN algorithm."""
    _label = 'flexible align - HetSIREN'
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

        group.addParam('inputVolume', params.PointerParam, allowsNull=True,
                       label="Starting volume", pointerClass='Volume',
                       help="If not provided, HetSIREN will learn from scratch how to recover heterogeneous states. If a "
                            "volume is provided, HetSIREN will use that volume as its starting state, and learn how to "
                            "modify it towards the heterogeneous states present in the data.\n\n"
                            "When should a map be provided?\n"
                            "    - If transport mass mode is set to Yes (check Advanced parameter in Reconstruction tab), it is "
                            "advisable to provide an input map\n"
                            "    - If transport mass mode is set to Yes and "
                            "If transport mass mode is set to No and local reconstruction is enabled (check Advanced "
                            "parameter in Reconstruction tab), it is highly recommended to provide a map\n\n"
                            "When is not advisable to provide a map?\n"
                            "If transport mass and local reconstruction are ser to No (check Advanced parameter in "
                            "Reconstruction tab), we recommend leaving this  parameter empty to give HetSIREN more freedom "
                            "to denoise and learn accurate states.")

        group.addParam('inputVolumeMask', params.PointerParam,
                       label="Reconstruction mask", pointerClass='VolumeMask', allowsNull=True,
                       help="Reconstruction mask meaning will depend on the mode chosen (NOTE: masks provided here MUST "
                            "be BINARY):\n\n"
                            "Transport mass is set to Yes\n"
                            "In this case, the provided mask will determine which voxels can be moved inside the volume box. "
                            "If you have provided an input volume, you can provide here a tight mask or a dilated tight mask, as HetSIREN "
                            "will be able to move those voxels anywhere needed to recover a given heterogeneous state. If don't provide an input "
                            "volume, we recommend providing a mask with enough mass to recover all expected states. A good rule of thumb "
                            "for this case is to provide a spherical mask with radius 0.25 * box_size.\n\n"
                            "Transport mass is set to No\n"
                            "If local reconstruction is set to No, we recommend to leave this parameter empty. If local reconstruction is "
                            "enabled, you should provide here a mask enclosing your region of interest AND any region where the motion of the "
                            "protein is expected to happen. Remember that, in Reconstruction mode (i.e., Transport mass is set to No), HetSIREN "
                            "will assume voxels as fixed in space. Any motions happening outside the binary mask will NOT be considered.")

        group.addParam('doDownsample', params.BooleanParam, default=False,
                       label='Downsample the images before training?',
                       help='If your GPU is not able to fit your particles in memory, you can play with this parameter to downsample your images. '
                            'In this way, you will be able to fit your particles at the expense of losing resolution in the volumes generated by the '
                            'network.')

        group.addParam('boxSize', params.IntParam, default=128,
                       condition="doDownsample",
                       label='Downsample particles to this box size',
                       help='If your GPU is not able to fit your particles in memory, you can play with this parameter to downsample your images. '
                            'In this way, you will be able to fit your particles at the expense of losing resolution in the volumes generated by the '
                            'network.')

        group.addParam('ctfType', params.EnumParam, choices=['None', 'Apply', 'Wiener', 'Precorrect', 'Premultiplied'],
                       default=1, label="CTF correction type",
                       display=params.EnumParam.DISPLAY_HLIST,
                       expertLevel=params.LEVEL_ADVANCED,
                       help="* *None*: CTF will not be considered\n"
                            "* *Apply*: CTF is applied to the projection generated from the reference map\n"
                            "* *Wiener*: input particle is CTF corrected by a Wiener fiter\n"
                            "* *Precorrect: similar to Wiener but CTF has already been corrected\n"
                            "* *Premultiplied*: input particles have already been multiplied by their CTF")

        group.addParam('particleDiameter', params.FloatParam, allowsNull=True,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Particle diameter (A)',
                       help="Particle diameter in Angstroms. It defines the background used to normalize the images and the "
                            "soft circular mask applied to them. If not provided, it is derived from the extent of the "
                            "reconstruction mask.")

        group.addParam('normalizeImages', params.BooleanParam, default=False,
                       label='Normalize images?',
                       help="Normalize each particle (zero mean and unit standard deviation in the background) before the "
                            "encoder, the losses and the contrast fit of the Gaussians.")

        form.addSection(label='Network')
        form.addParam('fineTune', params.BooleanParam, default=False, label='Fine tune previous network?',
                      help='When set to Yes, you will be able to provide a previously trained HetSIREN network to refine it with new '
                           'data. If set to No, you will train a new HetSIREN network from scratch.')

        form.addParam('hetsirenProtocol', params.PointerParam, condition='fineTune',
                      label="HetSIREN network to fine tune",
                      pointerClass='JaxProtFlexibleAlignmentHetSiren',
                      help="Previously executed 'flexible align - HetSIREN' whose network will be fine tuned.")

        form.addParam('refitReference', params.BooleanParam, default=False, condition='fineTune',
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Refit the reference?',
                      help="Fit the Gaussians to the starting volume (e.g. a MoDART map) and fine tune the reloaded network "
                           "on them. Needs a starting volume and a network trained with mass transport and an implicit or "
                           "point transformer decoder.")

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
        group.addParam('latDim', params.IntParam, default=8, label='Latent space dimension',
                       expertLevel=params.LEVEL_ADVANCED,
                       help="Dimension of the HetSIREN bottleneck (latent space dimension). It must be even, as the latent "
                            "is split into a motion half and an occupancy half.")

        group.addParam('epochs', params.IntParam, default=50,
                       label='Number of training epochs',
                       help="Number of epochs to train the network (i.e. how many times to loop over the whole dataset "
                             "of images - set to default to 50 - as a rule of thumb, consider 50 to 100 epochs enough "
                             "for 100k images / if your dataset is bigger or smaller, scale this value proportionally to it")

        group.addParam('autoBatchSize', params.BooleanParam, default=False,
                       label='Estimate batch size automatically?',
                       help="When set, HetSIREN estimates the largest batch size that fits on your GPU (this maximizes GPU "
                            "utilization, not necessarily accuracy).")

        group.addParam('batchSize', params.IntParam, default=8, label='Number of images in batch',
                       condition='not autoBatchSize',
                       help="Determines how many images will be load in the GPU at any moment during training (set by "
                            "default to 8 - you can control GPU memory usage easily by tuning this parameter to fit your "
                            "hardware requirements - we recommend using tools like nvidia-smi to monitor and/or measure "
                            "memory usage and adjust this value - keep also in mind that bigger batch sizes might be "
                            "less precise when looking for very local motions")

        group.addParam('learningRate', params.FloatParam, default=1e-4, label='Learning rate',
                       help="The learning rate (lr) sets the speed of learning. Think of the model as trying to find the "
                            "lowest point in a valley; the lr is the size of the step it takes on each attempt. A large "
                            "lr (e.g., 0.01) is like taking huge leaps — it's fast but can be unstable, overshoot the "
                            "lowest point, or cause NAN errors. A small lr (e.g., 1e-6) is like taking tiny shuffles — "
                            "it's stable but very slow and might get stuck before reaching the bottom. A good default is "
                            "often 0.0001. If training fails or errors explode, try making the lr 10 times smaller (e.g., "
                            "0.001 --> 0.0001).")

        group.addParam('lrSchedule', params.BooleanParam, default=False,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Use learning rate schedule?',
                       help="Use a linear warmup (10% of training) plus cosine-decay learning-rate schedule instead of a "
                            "constant learning rate.")

        group.addParam('encoderArch', params.EnumParam, choices=['convnn', 'mlpnn', 'resmlp'], default=2,
                       expertLevel=params.LEVEL_ADVANCED,
                       display=params.EnumParam.DISPLAY_HLIST,
                       label='Encoder architecture',
                       help="Encoder architecture mapping each image to its latent.\n"
                            "* *convnn*: convolutional encoder\n"
                            "* *mlpnn*: multilayer perceptron encoder\n"
                            "* *resmlp*: residual MLP with LayerNorm on the Fourier-cropped image and a bounded latent")

        group.addParam('gradClipNorm', params.FloatParam, default=1.0,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Gradient clipping norm',
                       help="Global-norm gradient clipping threshold. Set to 0 to disable.")

        group.addParam('emaDecay', params.FloatParam, default=0.0,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='EMA decay',
                       help="Decay of the exponential moving average of the weights (typical 0.999); the averaged weights "
                            "are exported as the final model. Set to 0 to disable.")

        group.addParam('datasetSplitFraction', params.StringParam, default='0.8,0.2',
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Training/validation split',
                       help="Fractions used to split the data into a training and a validation subset, following the format: "
                            "training_fraction,validation_fraction. The validation subset is only used to evaluate the network "
                            "and never updates its parameters. NOTE: both fractions must add up to one.")

        group = form.addGroup("Loss")
        group.addParam('lossType', params.EnumParam, choices=['mse', 'frc'], default=0,
                       expertLevel=params.LEVEL_ADVANCED,
                       display=params.EnumParam.DISPLAY_HLIST,
                       label='Reconstruction loss',
                       help="* *mse*: mean squared error in image space\n"
                            "* *frc*: Fourier ring correlation\n"
                            "The point transformer decoder always uses frc.")

        group.addParam('whitenLoss', params.BooleanParam, default=False,
                       expertLevel=params.LEVEL_ADVANCED,
                       condition='lossType == 0',
                       label='Whiten the reconstruction loss?',
                       help="Weight every Fourier ring of the reconstruction loss by the inverse of the residual power: "
                            "unweighted for the first 10 epochs, then updated every epoch from a running average of the "
                            "residual, keeping the overall loss scale.")

        group.addParam('perImageContrast', params.EnumParam, choices=['off', 'relative', 'absolute'], default=1,
                       expertLevel=params.LEVEL_ADVANCED,
                       display=params.EnumParam.DISPLAY_HLIST,
                       label='Per image contrast',
                       help="Fit a gray level per particle inside the reconstruction loss instead of relying on the global "
                            "contrast fixed before training.")

        group = form.addGroup("Latent space")
        group.addParam('klLambda', params.FloatParam, default=0.0,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='KL divergence weight',
                       help="Weight of the VAE KL divergence towards a standard normal prior. Set to 0 to disable the VAE and "
                            "use a deterministic latent.")

        group.addParam('klFreeBits', params.FloatParam, default=0.0,
                       expertLevel=params.LEVEL_ADVANCED,
                       condition='klLambda != 0',
                       label='KL free bits',
                       help="KL floor (nats per latent dimension) below which a dimension is not penalised, to avoid "
                            "posterior collapse. Set to 0 to get a plain KL.")

        group.addParam('nestedDropout', params.FloatParam, default=0.0,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Nested dropout',
                       help="Nested dropout on the decoded latent: each training sample keeps only its first k dimensions, "
                            "k ~ Geometric(p), so the dimensions are ordered by importance (PCA-like). Typical values are "
                            "0.1 to 0.5. Set to 0 to disable.")

        group.addParam('noiseInvarianceStrength', params.FloatParam, default=0.0,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Noise invariance strength',
                       help="Strength of the penalty 1 - correlation, per latent dimension, between the latents of the image "
                            "and of the image with a second draw of noise coloured like its background. After the first "
                            "epoch, its weight is recalibrated every epoch so its encoder gradient is this fraction of the "
                            "reconstruction one. Typical values are 0.1 to 0.5. Set to 0 to disable.")

        group.addParam('decouplingLambda', params.FloatParam, default=1e-4,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Decoupling weight',
                       help="Weight of the decoupling regularization that makes the latent invariant to pose and CTF.")

        group = form.addGroup("Network regularization")
        group.addParam('denoisingStrength', params.FloatParam, default=1e-4, label='Denoising strength',
                       expertLevel=params.LEVEL_ADVANCED,
                       help="Determines how strongly HetSIREN will learn to remove noise from the resulting volumes. "
                            "Increasing the value of this parameter will result in a stronger regularization of the noise,"
                            " but it may affect the protein signal as well. (NOTE: We recommend setting this parameter in "
                            "the range 0.0001 to 0.1)")

        group.addParam('distancePreservationLambda', params.FloatParam, default=1e-4,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Distance preservation weight',
                       help="Weight of the local distance preservation regularization between the deformed and the mean "
                            "state. Higher values give stiffer deformations.")

        group.addParam('geometricLambda', params.FloatParam, default=0.0,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Geometric correction weight',
                       help="Weight of the geometric-correction loss on the decoder Jacobian (Gram determinant), pushing "
                            "towards a locally volume-preserving latent-to-shape map. Expensive. Set to 0 to disable.")

        group = form.addGroup("Logging")
        group.addParam('logImagesEvery', params.IntParam, default=1,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Log images every (epochs)',
                       help="How often (in epochs) to log the cheap intermediate results to Tensorboard: the predicted images "
                            "and the central slices of the predicted volumes. Set to 0 to disable.")

        group.addParam('logLandscapeEvery', params.IntParam, default=5,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Log landscape every (epochs)',
                       help="How often (in epochs) to log the expensive intermediate results: the latent space embedding "
                            "(Tensorboard projector), the intermediate volumes written to disk and the angular distribution "
                            "plots. Set to 0 to disable. NOTE: the embedding alone costs several seconds per call, so on "
                            "small datasets it can dominate the training time.")

        group.addParam('logCheckpointEvery', params.IntParam, default=5,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Save checkpoint every (epochs)',
                       help="How often (in epochs) to write the intermediate checkpoint used to resume an interrupted "
                            "training. Set to 0 to disable.")

        group.addParam('logTimeBudget', params.FloatParam, default=0.0,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Logging time budget',
                       help="Maximum fraction of the total wall clock that may be spent logging (e.g. 0.05 for 5%). When "
                            "exceeded, the expensive logging is skipped until training catches up. Set to 0 to disable.")

        group.addParam('logSync', params.BooleanParam, default=False,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Log on the training thread?',
                       help="By default, logging runs on a background thread so it overlaps with the next epoch. When set, "
                            "logs are written in lockstep with training (e.g. while debugging).")

        form.addSection(label='Reconstruction')
        form.addParam('massTransport', params.BooleanParam, default=False,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Mass transportation',
                      help='When set, HetSIREN will be able to "move" the mass inside the mask '
                           'instead of just reconstructing the volume. This implies that HetSIREN will estimate the motion '
                           'to be applied to the points within the provided mask, instead of considering them fixed in space. '
                           'This approach is useful when working with large box sizes that '
                           'do not fit in GPU memory, or when a more through analysis of motions is desired. '
                           'If False, HetSIREN program will perform heterogeneous reconstruction.')

        form.addParam('isImplicit', params.BooleanParam, expertLevel=params.LEVEL_ADVANCED, default=False,
                      condition='massTransport',
                      label='Use implicit architecture?',
                      help='HetSIREN implicit architecture is able to report better local motions present in the protein at the '
                           'expense of a larger GPU memory consumption. If you are running out of memory in the GPU, setting this '
                           'parameter to False might help reducing the memory burden.')

        form.addParam('pointTransformer', params.BooleanParam, expertLevel=params.LEVEL_ADVANCED, default=False,
                      condition='massTransport and not isImplicit',
                      label='Use point transformer architecture?',
                      help='Use a point transformer decoder on the Gaussian point cloud.')

        form.addParam('filmHeads', params.BooleanParam, expertLevel=params.LEVEL_ADVANCED, default=False,
                      condition='massTransport and (isImplicit or pointTransformer)',
                      label='Use FiLM heads?',
                      help='Replace the implicit SIREN heads (and the point transformer occupancy head) by a coordinate SIREN '
                           'modulated by the latent through FiLM layers.')

        form.addParam('motionOnly', params.BooleanParam, expertLevel=params.LEVEL_ADVANCED, default=False,
                      condition='massTransport',
                      label='Estimate motion only?',
                      help='Estimate conformational changes only: the amplitudes stay at their reference values.')

        form.addParam('trainInverse', params.BooleanParam, expertLevel=params.LEVEL_ADVANCED, default=False,
                      condition='massTransport',
                      label='Train inverse decoder?',
                      help='Additionally train an inverse decoder mapping the deformed point cloud back to the latent.')

        form.addParam('reconSplitFinal', params.FloatParam, expertLevel=params.LEVEL_ADVANCED, default=0.1,
                      condition='massTransport',
                      label='Amplitude term weight',
                      help='Amplitude-term weight of the reconstruction loss, from the first step. The loss is '
                           'w x L(learned amplitudes, frozen coords) + (1-w) x L(reference amplitudes, learned coords).')

        form.addParam('deformationStrength', params.FloatParam, expertLevel=params.LEVEL_ADVANCED, default=0.3,
                      condition='massTransport and inputVolume',
                      label='Deformation prior strength',
                      help='Strength of the graph deformation priors (relative edge strain + repulsion) with respect to the '
                           'reconstruction loss: after the first epoch, the prior weight is recalibrated every epoch so its '
                           'gradient on the decoder is this fraction of the reconstruction gradient. Lower for larger motions, '
                           'higher for stiffer ones. Set to 0 to disable.')

        form.addParam('occupancyStrength', params.FloatParam, expertLevel=params.LEVEL_ADVANCED, default=0.0,
                      condition='massTransport and inputVolume',
                      label='Occupancy prior strength',
                      help='Weight of the occupancy prior on the per-particle amplitude change: an L1 shrinkage (stronger '
                           'where the reference is empty) plus a graph total variation favouring spatially coherent '
                           'occupancy, both relative to the mean reference density. Set to 0 to disable.')

        form.addParam('occupancyWarmupEpochs', params.FloatParam, expertLevel=params.LEVEL_ADVANCED, default=3.0,
                      condition='massTransport',
                      label='Occupancy warmup epochs',
                      help='Epochs over which the per-particle amplitude change is ramped in, so motion is learned before '
                           'occupancy. Set to 0 to disable the ramp.')

        form.addParam('learnReferenceEpochs', params.IntParam, expertLevel=params.LEVEL_ADVANCED, default=0,
                      condition='massTransport and isImplicit',
                      label='Learn reference epochs',
                      help='Learn the positions of the reference Gaussians during the first N epochs, then freeze them. '
                           'The learned reference is saved as an output volume. Set to 0 to disable.')

        form.addParam('referenceLr', params.FloatParam, expertLevel=params.LEVEL_ADVANCED, allowsNull=True,
                      condition='massTransport and isImplicit and learnReferenceEpochs > 0',
                      label='Reference learning rate',
                      help='Step size used to learn the reference positions. If not provided: '
                           '2e-4 * sqrt(batch size / 128).')

        form.addParam('numberGaussians', params.IntParam, label='Number of gaussians (fixed)',
                      expertLevel=params.LEVEL_ADVANCED,
                      allowsNull=True, condition="inputVolume",
                      help="Before training the network, HetSIREN will try to fit a set of Gaussians in the reference volume to recreate it. "
                           "The default criterium is to automatically determine the number of Gaussians neede to reproduce the reference volume "
                           "with high-fidelity. However, if you prefer to fix the number of Gaussians in advance based on your own criterium (e.g., "
                           "the number of residues in your protein), you can set this parameter. When set, the HetSIREN will fit this fixed number of Gaussians "
                           "so that the reproduce the reference volume as well as possible.")

        form.addParam('fitResolution', params.FloatParam, label='Gaussian fit resolution (A)',
                      expertLevel=params.LEVEL_ADVANCED,
                      allowsNull=True, condition="inputVolume",
                      help="Resolution (in Angstroms) the automatic Gaussian fit has to reproduce. If not provided, the number "
                           "of Gaussians is grown until the fit reaches FSC >= 0.5 on every populated shell.")

        form.addParam('latticeStride', params.IntParam, expertLevel=params.LEVEL_ADVANCED, default=1,
                      condition='not massTransport or not inputVolume',
                      label='Lattice stride',
                      help='Keep one fixed-lattice point every this many voxels along each axis (trades resolution for memory).')

        form.addParam('bilinearScatter', params.BooleanParam, expertLevel=params.LEVEL_ADVANCED, default=False,
                      condition='not massTransport',
                      label='Bilinear scatter?',
                      help='Spread each fixed-lattice point over its four neighbouring pixels instead of the nearest one.')

        form.addParam('localRecon', params.BooleanParam, expertLevel=params.LEVEL_ADVANCED, default=False,
                       condition='not massTransport',
                       label='Local reconstruction',
                       help='When set, HetSIREN will turn to local heterogeneous reconstruction/refinement mod, '
                            'focusing the analysis of heterogeneity to a region of interest enclosed by the provided reference mask.')

        group = form.addGroup("Pose refinement")
        group.addParam('rigidGauge', params.EnumParam, choices=['core', 'mass', 'off'], default=2,
                       expertLevel=params.LEVEL_ADVANCED,
                       display=params.EnumParam.DISPLAY_HLIST,
                       label='Rigid gauge',
                       help="Remove the global rigid motion from the decoder output so pose errors go to the pose head. "
                            "Without mass transport, it requires a reference volume.\n"
                            "* *core*: frame fitted on the least-moving points\n"
                            "* *mass*: mass-weighted fit\n"
                            "* *off*: diagnostic only")

        group.addParam('rigidDriftLambda', params.FloatParam, default=1.0,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Rigid drift weight',
                       help="Weight of the penalty on the rigid motion the gauge removes from the decoder displacement field.")

        group.addParam('poseRefineReg', params.FloatParam, default=0.1,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Pose refinement penalty',
                       help="Weight of the hinge penalising pose refinements beyond the maximum free rotation. Set to 0 to "
                            "disable.")

        group.addParam('poseRefineMaxAngle', params.FloatParam, default=8.0,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Maximum free rotation (deg)',
                       help="Rotation (degrees) the pose refinement may apply freely; the excess is penalised with the pose "
                            "refinement penalty.")

        group.addParam('shiftRefineReg', params.FloatParam, default=0.1,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Shift refinement penalty',
                       help="Weight of the hinge penalising shift refinements beyond the maximum free shift. Set to 0 to "
                            "disable.")

        group.addParam('shiftRefineMax', params.FloatParam, default=4.0,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Maximum free shift (px)',
                       help="Shift (pixels) the refinement may apply freely; the excess is penalised with the shift "
                            "refinement penalty.")

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
        self._insertFunctionStep(self.trainingPredictStep)
        self._insertFunctionStep(self.createOutputStep)

    def writeMetaDataStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnVolMask = self._getFileName('fnVolMask')

        inputParticles = self.inputParticles.get()
        Xdim = inputParticles.getXDim()
        newXdim = self.boxSize.get() if self.doDownsample.get() else Xdim
        vol_mask_dim = newXdim

        if self.inputVolume.get():
            ih = ImageHandler()
            inputVolume = self.inputVolume.get().getFileName()
            ih.convert(self._getXmippFileName(inputVolume), fnVol)
            curr_vol_dim = ImageHandler(self._getXmippFileName(inputVolume)).getDimensions()[-1]
            if curr_vol_dim != vol_mask_dim:
                self.runJob("xmipp_image_resize",
                            "-i %s --fourier %d " % (fnVol, vol_mask_dim), numberOfMpi=1,
                            env=xmipp3.Plugin.getEnviron())
            ih.setSamplingRate(fnVol, inputParticles.getSamplingRate())

        if self.inputVolumeMask.get():
            ih = ImageHandler()
            inputMask = self.inputVolumeMask.get().getFileName()
            if inputMask:
                ih.convert(self._getXmippFileName(inputMask), fnVolMask)
                curr_mask_dim = ImageHandler(self._getXmippFileName(inputMask)).getDimensions()[-1]
                if curr_mask_dim != vol_mask_dim:
                    self.runJob("xmipp_image_resize",
                                "-i %s --dim %d --interp nearest" % (fnVolMask, vol_mask_dim), numberOfMpi=1,
                                env=xmipp3.Plugin.getEnviron())
        else:
            ImageHandler().createCircularMask(fnVolMask, boxSize=vol_mask_dim, is3D=True)

        writeSetOfParticles(inputParticles, imgsFn)

        # Write extra attributes (if needed)
        md = XmippMetaData(imgsFn)
        if hasattr(inputParticles.getFirstItem(), "_xmipp_subtomo_labels"):
            labels = np.asarray([int(particle._xmipp_subtomo_labels) for particle in inputParticles.iterItems()])
            md[:, "subtomo_labels"] = labels
        md.write(imgsFn, overwrite=True)

        if newXdim != Xdim:
            params = "-i %s -o %s --save_metadata_stack %s --fourier %d" % \
                     (imgsFn,
                      self._getTmpPath('scaled_particles.stk'),
                      self._getExtraPath('scaled_particles.xmd'),
                      newXdim)
            if self.numberOfMpi.get() > 1:
                params += " --mpi_job_size %d" % int(inputParticles.getSize() / self.numberOfMpi.get())
            self.runJob("xmipp_image_resize", params, numberOfMpi=self.numberOfMpi.get(),
                        env=xmipp3.Plugin.getEnviron())
            moveFile(self._getExtraPath('scaled_particles.xmd'), imgsFn)

    def trainingPredictStep(self):
        md_file = self._getFileName('imgsFn')
        vol_file = self._getFileName('fnVol')
        mask_file = self._getFileName('fnVolMask')
        out_path = self._getExtraPath()
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        batch_size = "auto" if self.autoBatchSize.get() else self.batchSize.get()
        learningRate = self.learningRate.get()
        epochs = self.epochs.get()
        latDim = self.latDim.get()
        denoisingStrength = self.denoisingStrength.get()
        newXdim = self.boxSize.get() if self.doDownsample.get() else self.inputParticles.get().getXDim()
        correctionFactor = self.inputParticles.get().getXDim() / newXdim
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        args = "--md %s --sr %f --lat_dim %d --epochs %d --batch_size %s --learning_rate %s --output_path %s --denoising_strength %f " \
               % (md_file, sr, latDim, epochs, batch_size, learningRate, out_path, denoisingStrength)

        args += "--dataset_split_fraction %s --encoder_arch %s --grad_clip_norm %s --ema_decay %s " \
                "--loss_type %s --per_image_contrast %s --kl_lambda %s --kl_free_bits %s --nested_dropout %s " \
                "--noise_invariance_strength %s --decoupling_lambda %s --distance_preservation_lambda %s " \
                "--geometric_lambda %s --lattice_stride %d --rigid_gauge %s --rigid_drift_lambda %s " \
                "--pose_refine_reg %s --pose_refine_max_angle %s --shift_refine_reg %s --shift_refine_max %s " \
                "--log_images_every %d --log_landscape_every %d --log_checkpoint_every %d --log_time_budget %s " \
                % (self.datasetSplitFraction.get().replace(" ", ""), self.getEnumText('encoderArch'),
                   self.gradClipNorm.get(), self.emaDecay.get(), self.getEnumText('lossType'),
                   self.getEnumText('perImageContrast'), self.klLambda.get(), self.klFreeBits.get(),
                   self.nestedDropout.get(), self.noiseInvarianceStrength.get(), self.decouplingLambda.get(),
                   self.distancePreservationLambda.get(), self.geometricLambda.get(), self.latticeStride.get(),
                   self.getEnumText('rigidGauge'), self.rigidDriftLambda.get(), self.poseRefineReg.get(),
                   self.poseRefineMaxAngle.get(), self.shiftRefineReg.get(), self.shiftRefineMax.get(),
                   self.logImagesEvery.get(), self.logLandscapeEvery.get(), self.logCheckpointEvery.get(),
                   self.logTimeBudget.get())

        if self.inputVolume.get():
            args += '--vol %s ' % vol_file

            if self.numberGaussians.get() is not None:
                args += '--num_gaussians %d ' % self.numberGaussians.get()

            if self.fitResolution.get() is not None:
                args += '--fit_resolution %s ' % self.fitResolution.get()

        if self.inputVolumeMask.get():
            args += '--mask %s ' % mask_file

        if self.ctfType != 0:
            if self.ctfType.get() == 1:
                args += '--ctf_type apply '
            elif self.ctfType.get() == 2:
                args += '--ctf_type wiener '
            elif self.ctfType.get() == 3:
                args += '--ctf_type precorrect '
            elif self.ctfType.get() == 4:
                args += '--ctf_type premultiplied '
        else:
            args += '--ctf_type None '

        if self.particleDiameter.get() is not None:
            args += '--particle_diameter %s ' % self.particleDiameter.get()

        if self.normalizeImages:
            args += '--normalize_images '

        if self.lrSchedule:
            args += '--lr_schedule '

        if self.whitenLoss and self.lossType.get() == 0:
            args += '--whiten_loss '

        if self.logSync:
            args += '--log_sync '

        if self.lazyLoad:
            args += '--load_images_to_ram '
        else:
            if self.scratchFolder.get() is not None:
                args += '--ssd_scratch_folder %s ' % self.scratchFolder.get()

        if self.massTransport:
            args += '--transport_mass --recon_split_final %s --occupancy_warmup_epochs %s ' \
                    % (self.reconSplitFinal.get(), self.occupancyWarmupEpochs.get())

            if self.inputVolume.get():
                args += '--deformation_strength %s --occupancy_strength %s ' \
                        % (self.deformationStrength.get(), self.occupancyStrength.get())

            if self.isImplicit:
                args += '--implicit_network '

                if self.learnReferenceEpochs.get() > 0:
                    args += '--learn_reference_epochs %d ' % self.learnReferenceEpochs.get()
                    if self.referenceLr.get() is not None:
                        args += '--reference_lr %s ' % self.referenceLr.get()
            elif self.pointTransformer:
                args += '--point_transformer '

            if self.filmHeads and (self.isImplicit or self.pointTransformer):
                args += '--film_heads '

            if self.motionOnly:
                args += '--motion_only '

            if self.trainInverse:
                args += '--train_inverse '

        else:
            if self.bilinearScatter:
                args += '--bilinear_scatter '

            if self.localRecon:
                args += '--local_reconstruction '

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        program = hax.Plugin.getProgram("hetsiren", gpu)
        if not os.path.isdir(self._getExtraPath("HetSIREN")):
            train_args = args + '--mode train '
            if self.fineTune:
                train_args += '--reload %s ' % self.hetsirenProtocol.get()._getExtraPath()
                if self.refitReference:
                    train_args += '--refit_reference '
            self.runJob(program, train_args, numberOfMpi=1)
        self.runJob(program, args + f'--mode predict --reload {self._getExtraPath()}', numberOfMpi=1)

    def createOutputStep(self):
        out_path_vols = self._getExtraPath('volumes')
        model_path = self._getExtraPath('HetSIREN')
        md_file = self._getFileName('predictedFn')
        out_path = self._getExtraPath()
        Xdim = self.inputParticles.get().getXDim()
        self.newXdim = self.boxSize.get()

        metadata = XmippMetaData(md_file)
        correctionFactor = Xdim / self.newXdim
        latent_space = np.asarray([np.fromstring(item, sep=',') for item in metadata[:, 'latent_space']])
        euler_rot = metadata[:, 'angleRot']
        euler_tilt = metadata[:, 'angleTilt']
        euler_psi = metadata[:, 'anglePsi']
        shift_x = correctionFactor * metadata[:, 'shiftX']
        shift_y = correctionFactor * metadata[:, 'shiftY']

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticlesFlex(progName=const.HETSIREN)

        partSet.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())
        partSet.setAlignmentProj()
        inverseTransform = partSet.getAlignment() == ALIGN_PROJ

        idx = 0
        for particle in inputSet.iterItems():
            outParticle = ParticleFlex(progName=const.HETSIREN)
            outParticle.copyInfo(particle)
            outParticle.setZFlex(latent_space[idx])

            # Set new transformation matrix
            tr = matrixFromGeometry(np.array([shift_x[idx], shift_y[idx], 0.0]),
                                    np.array([euler_rot[idx], euler_tilt[idx], euler_psi[idx]]),
                                    inverseTransform)
            outParticle.getTransform().setMatrix(tr)

            partSet.append(outParticle)
            idx += 1

        partSet.getFlexInfo().modelPath = String(model_path)

        if self.inputVolume.get():
            inputVolume = self.inputVolume.get().getFileName()
            partSet.getFlexInfo().refMap = String(inputVolume)

        if self.inputVolumeMask.get():
            inputMask = self.inputVolumeMask.get().getFileName()
            partSet.refMask = String(inputMask)

        if self.ctfType.get() != 0:
            if self.ctfType.get() == 1:
                partSet.getFlexInfo().ctfType = String("apply")
            elif self.ctfType.get() == 2:
                partSet.getFlexInfo().ctfType = String("wiener")
            elif self.ctfType.get() == 3:
                partSet.getFlexInfo().ctfType = String("precorrect")
            elif self.ctfType.get() == 4:
                partSet.getFlexInfo().ctfType = String("premultiplied")

        if self.useGpu.get():
            gpu = str(self.getGpuList()[0])
        else:
            gpu = None

        latents_kmeans = KMeans(n_clusters=20).fit(latent_space).cluster_centers_

        latents_file_txt = os.path.join(out_path, 'latents.txt')
        np.savetxt(latents_file_txt, latents_kmeans)

        if not os.path.isdir(out_path_vols):
            makePath(out_path_vols)

        args = "--latents_file %s --output_path %s --reload %s" % (latents_file_txt, out_path_vols, model_path)
        program = hax.Plugin.getProgram("decode_states_from_latents", gpu)
        self.runJob(program, args, numberOfMpi=1)

        outVols = self._createSetOfVolumes()
        outVols.setSamplingRate(inputSet.getSamplingRate())
        for idx in range(latents_kmeans.shape[0]):
            outVol = Volume()
            outVol.setSamplingRate(inputSet.getSamplingRate())

            ImageHandler().scaleSplines(os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"),
                                        os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"),
                                        finalDimension=inputSet.getXDim(), overwrite=True)

            ImageHandler().setSamplingRate(os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"),
                                           inputSet.getSamplingRate())

            outVol.setLocation(os.path.join(out_path_vols, f"decoded_volume_{idx:04d}.mrc"))
            outVols.append(outVol)

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(inputSet, partSet)

        self._defineOutputs(outputVolumes=outVols)
        self._defineTransformRelation(inputSet, outVols)

        learned_ref = self._getExtraPath('learned_reference.mrc')
        if os.path.isfile(learned_ref):
            ImageHandler().scaleSplines(learned_ref, learned_ref, finalDimension=inputSet.getXDim(), overwrite=True)
            ImageHandler().setSamplingRate(learned_ref, inputSet.getSamplingRate())
            refVol = Volume()
            refVol.setSamplingRate(inputSet.getSamplingRate())
            refVol.setLocation(learned_ref)
            self._defineOutputs(outputLearnedReference=refVol)
            self._defineSourceRelation(inputSet, refVol)

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

        if self.latDim.get() % 2 != 0:
            errors.append("Latent space dimension must be even")

        if self.fineTune and self.refitReference:
            prot = self.hetsirenProtocol.get()
            if not self.inputVolume.get():
                errors.append("Refitting the reference needs a starting volume")
            if prot is not None and not (prot.massTransport.get() and (prot.isImplicit.get() or prot.pointTransformer.get())):
                errors.append("Refitting the reference needs a network trained with mass transport and an implicit or "
                              "point transformer decoder")

        return errors

    def _warnings(self):
        warnings = []

        return warnings

    # --------------------------- UTILS functions -----------------------

    def _getXmippFileName(self, filename):
        if getExt(filename) == ".mrc":
            filename += ":mrc"
        return filename