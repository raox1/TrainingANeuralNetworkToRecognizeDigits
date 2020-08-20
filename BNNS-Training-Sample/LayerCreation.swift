/*
See LICENSE folder for this sample’s licensing information.

Abstract:
BNNS layer creation extension.
*/

import Accelerate

extension TrainingSample {
    static var fusedConvBatchNormLayer: BNNS.FusedLayer = {
        guard let fusedConvBatchNormLayer = BNNS.FusedConvolutionNormalizationLayer(
                input: inputArray,
                output: batchNormOutputArray,
                convolutionWeights: convolutionWeightsArray,
                convolutionBias: convolutionBiasArray,
                convolutionStride: (1, 1),
                convolutionDilationStride: (1, 1),
                convolutionPadding: .symmetric(x: convolutionPadding,
                                               y: convolutionPadding),
                normalization: .batch(movingMean: batchNormMovingMeanArray,
                                      movingVariance: batchNormMovingVarianceArray),
                normalizationBeta: batchNormBetaArray,
                normalizationGamma: batchNormGammaArray,
                normalizationMomentum: 0.9,
                normalizationEpsilon: 1e-07,
                normalizationActivation: .rectifiedLinear,
                filterParameters: filterParameters) else {
            fatalError("Unable to create `fusedConvBatchNormLayer`.")
        }
        
        return fusedConvBatchNormLayer
    }()
    
    static var poolingLayer: BNNS.PoolingLayer = {
        guard let poolingLayer = BNNS.PoolingLayer(
                type: .max(xDilationStride: 1, yDilationStride: 1),
                input: batchNormOutputArray,
                output: poolingOutputArray,
                bias: nil,
                activation: .identity,
                kernelSize: (2, 2),
                stride: (2, 2),
                padding: .zero,
                filterParameters: filterParameters) else {
            fatalError("Unable to create `poolingLayer`.")
        }
        
        return poolingLayer
    }()
    
    static var fullyConnectedLayer: BNNS.FullyConnectedLayer = {
        
        let desc = BNNSNDArrayDescriptor(dataType: .float,
                                         shape: .vector(poolingOutputSize))
        
        guard let fullyConnectedLayer = BNNS.FullyConnectedLayer(
                input: desc,
                output: fullyConnectedOutputArray,
                weights: fullyConnectedWeightsArray,
                bias: nil,
                activation: .identity,
                filterParameters: filterParameters) else {
            fatalError("Unable to create `fullyConnectedLayer`.")
        }
        
        return fullyConnectedLayer
    }()
    
    static var lossLayer: BNNS.LossLayer = {
        
        guard let lossLayer = BNNS.LossLayer(input: fullyConnectedOutputArray,
                                             output: lossOutputArray,
                                             lossFunction: .softmaxCrossEntropy(labelSmoothing: 0),
                                             lossReduction: .reductionMean,
                                             filterParameters: filterParameters) else {
            fatalError("Unable to create `lossLayer`.")
        }
        
        return lossLayer
    }()
}
