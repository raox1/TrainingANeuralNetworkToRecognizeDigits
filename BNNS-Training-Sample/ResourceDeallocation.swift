/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Resource deallocation extension.
*/

import Accelerate

extension TrainingSample {
    // Deallocate memory allocated to the array descriptors.
    static func freeArrayDescriptors() {
        oneHotLabels.deallocate()
        inputArray.deallocate()
        convolutionWeightsArray.deallocate()
        convolutionBiasArray.deallocate()
        batchNormOutputArray.deallocate()
        batchNormBetaArray.deallocate()
        batchNormGammaArray.deallocate()
        batchNormMovingVarianceArray.deallocate()
        batchNormMovingMeanArray.deallocate()
        poolingOutputArray.deallocate()
        fullyConnectedWeightsArray.deallocate()
        fullyConnectedOutputArray.deallocate()
        lossOutputArray.deallocate()
        lossInputGradientArray.deallocate()
        fullyConnectedInputGradientArray.deallocate()
        fullyConnectedWeightGradientArray.deallocate()
        fullyConnectedWeightAccumulator1.deallocate()
        fullyConnectedWeightAccumulator2.deallocate()
        convolutionInputGradientArray.deallocate()
        convolutionWeightGradientArray.deallocate()
        convolutionBiasGradientArray.deallocate()
        batchNormBetaGradientArray.deallocate()
        batchNormGammaGradientArray.deallocate()
        convolutionWeightAccumulator1.deallocate()
        convolutionWeightAccumulator2.deallocate()
        convolutionBiasAccumulator1.deallocate()
        convolutionBiasAccumulator2.deallocate()
        batchNormBetaAccumulator1.deallocate()
        batchNormBetaAccumulator2.deallocate()
        batchNormGammaAccumulator1.deallocate()
        batchNormGammaAccumulator2.deallocate()
        poolingInputGradientArray.deallocate()
    }
}
