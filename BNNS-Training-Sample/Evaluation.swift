/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Network evaluation extension.
*/

import Accelerate

extension TrainingSample {
    
    static func test() -> Float {
        
        generateInputAndLabels()
        
        // Perform the forward pass.
        do {
            try fusedConvBatchNormLayer.apply(batchSize: batchSize,
                                              input: inputArray,
                                              output: batchNormOutputArray,
                                              for: .inference)
            
            try poolingLayer.apply(batchSize: batchSize,
                                   input: batchNormOutputArray,
                                   output: poolingOutputArray)
            
            try fullyConnectedLayer.apply(batchSize: batchSize,
                                          input: poolingOutputArray,
                                          output: fullyConnectedOutputArray)
        } catch {
            fatalError("Test forward pass failed.")
        }
        
        // Calculate the accuracy of the model.
        guard
            let fullyConnected = fullyConnectedOutputArray.makeArray(
                of: Float.self,
                batchSize: batchSize),
            let labels = oneHotLabels.makeArray(
                of: Float.self,
                batchSize: batchSize) else {
            fatalError("Unable to create arrays for evaluation.")
        }
        
        var correctCount = 0
        
        for sample in 0 ..< batchSize {
            let offset = fullyConnectedOutputWidth * sample
            
            let fullyConnectedBatch = fullyConnected[offset ..< offset + fullyConnectedOutputWidth]
            let predictedDigit = vDSP.indexOfMaximum(fullyConnectedBatch).0
            
            let oneHotLabelsBatch = labels[offset ..< offset + fullyConnectedOutputWidth]
            let label = vDSP.indexOfMaximum(oneHotLabelsBatch).0
            
            if label == predictedDigit {
                correctCount += 1
            }
            
            print("Sample \(sample) — digit: \(label) | prediction: \(predictedDigit)")
        }
        
        // Return the accuracy as a percentage.
        let score = 100 * (Float(correctCount) / Float(batchSize))
        
        return score
    }
}
