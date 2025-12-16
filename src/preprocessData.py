import os

import numpy as np
import pandas as pd

np.random.seed(42)

def filterHighlights(allReplayData, K=20, threshold=0.5):
    allGTHighlights = []

    for replayData in allReplayData:

        replayData[0] = replayData[1]

        replayData = [(replayData[i], i) for i in range(len(replayData))]
        replayData.sort(reverse=True, key=lambda x:x[0])

        gtHighlights = []

        index = 0
        while (replayData[index][0] > threshold and index < K):
            gtHighlights.append(replayData[index][1])
            index += 1

        allGTHighlights.append(gtHighlights)

    return allGTHighlights

def calculateMetrics(predicted, groundTruth):
    P = set(predicted)
    G = set(groundTruth)
    
    # Number of correctly predicted highlights
    hits = len(P & G)
    
    # Hit rate: 1 if at least one correct prediction, 0 otherwise
    hitRate = 1.0 if hits > 0 else 0.0
    
    # Precision: fraction of predictions that are correct
    precision = hits / len(P) if len(P) > 0 else 0.0
    
    # Recall: fraction of ground truth that are predicted
    recall = hits / len(G) if len(G) > 0 else 0.0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'hit_rate': hitRate,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def evaluateRandomBaseline(allGTHighlights, K):
    allMetrics = {
        'hit_rate': [],
        'precision': [],
        'recall': [],
        'f1': [],
    }
    
    for i, gtHighlights in enumerate(allGTHighlights):
        
        predicted = np.random.choice(100, size=K, replace=False).tolist()
        
        metrics = calculateMetrics(predicted, gtHighlights)
        
        for key in allMetrics:
            allMetrics[key].append(metrics[key])
    
    avgMetrics = {key: np.mean(values) for key, values in allMetrics.items()}
    
    return avgMetrics

def main():
    projectDir = os.getcwd()

    dataDir = os.path.join(os.path.dirname(projectDir), 'rhapsody', 'data')

    testFilePath = os.path.join(dataDir, 'test-00000-of-00001.parquet')
    
    # Read the parquet file into a DataFrame
    df = pd.read_parquet(testFilePath)
    
    allReplayData = df['gt'].to_numpy()
    allGTHighlights = filterHighlights(allReplayData)

    numGTHighlights = []
    for gtHighlight in allGTHighlights:
        numGTHighlights.append(len(gtHighlight))

    print(f"average number of highlights: {np.mean(numGTHighlights)}")
    print(f"std of highlights: {np.std(numGTHighlights)}")
    #print(len(allGTHighlights))

    print("\nRandom Sampling Baseline Results, K=5:")
    avgMetrics = evaluateRandomBaseline(allGTHighlights, K=5)
    for metric, value in avgMetrics.items():
        print(f"{metric}: {value:.3f}")

    print("\nRandom Sampling Baseline Results, K=20:")
    avgMetrics = evaluateRandomBaseline(allGTHighlights, K=20)
    for metric, value in avgMetrics.items():
        print(f"{metric}: {value:.3f}")
    



if __name__ == "__main__":
    main()