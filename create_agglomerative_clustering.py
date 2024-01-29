import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import time
import argparse
import dill as pickle
from memory_profiler import profile


print("USAGE: create_agglomerative_clustering.py -p <POINT_FILE> -v <VOCAB_FILE> -k <CLUSTERS> -o <OUTPUT_FOLDER>")


@profile
def agglomerative_cluster(points, vocab, K, output_path, ref=''):
    """
    Uses the point.npy, vocab.npy files of a layer (generated using https://github.com/hsajjad/ConceptX/ library) to produce a clustering of <K> clusters at <output_path> named clusters-agg-{K}.txt

    """
    print('Starting agglomerative clustering...')
    clustering = AgglomerativeClustering(n_clusters=K,compute_distances=True).fit(points)
    print('Finished clustering')
    fn = f"{output_path}/model-{K}-agglomerative-clustering{ref}.pkl"
    with open(fn, "wb") as fp:
        pickle.dump(clustering,fp)

    clusters = defaultdict(list)
    for i,label in enumerate(clustering.labels_):
        clusters[clustering.labels_[i]].append(vocab[i])

    # Write Clusters in the format (Word|||WordID|||SentID|||TokenID|||ClusterID)
    out = ""
    for key in clusters.keys():
        for word in clusters[key]:
            out += word+"|||"+str(key)+"\n"
    
    with open(f"{output_path}/clusters-agg-{K}{ref}.txt",'w') as f:
        f.write(output)

    return out


parser = argparse.ArgumentParser()
parser.add_argument("--vocab-file","-v", help="output vocab file with complete path")
parser.add_argument("--point-file","-p", help="output point file with complete path")
parser.add_argument("--output-path","-o", help="output path clustering model and result files")
parser.add_argument("--cluster","-k", help="cluster number")
parser.add_argument("--count","-c", help="point count ratio", default=-1)


args2 = parser.parse_args()
vocab_file = args2.vocab_file
point_file = args2.point_file
output_path = args2.output_path
point_count_ratio = float(args2.count)
K = int(args2.cluster)


vocab = np.load(vocab_file)
original_count = len(vocab)
useable_count = int(point_count_ratio*original_count) if point_count_ratio != -1 else -1
vocab = np.load(vocab_file)[:useable_count]

points = np.load(point_file)[:useable_count, :]
			
start_time = time.time()
ref = '-' + str(point_count_ratio) if point_count_ratio > 0 else ''
output = agglomerative_cluster(points, vocab, K, output_path, ref)
end_time = time.time()

print(f"Runtime: {end_time - start_time}")
