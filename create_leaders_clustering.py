import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import argparse
import time 
from memory_profiler import profile
from annoy import AnnoyIndex
import statistics


print("USAGE: create_leaders_clustering.py -p <POINT_FILE> -v <VOCAB_FILE> -k <CLUSTERS> -o <OUTPUT_FOLDER> -t <TAU> --fast ")


class Clique:
    """
    A clique of follower points for a leader point
    """
    def __init__(self, p, j):
        """
        Initialize a clique by adding the leader and its index
        """
        self.members= [p]
        self.member_indices = [j]
        self.centroid = p

    def __len__(self):
        return len(self.members)

    def add(self, p, j):
        """
        Add a new follower to the clique and update the centroid
        """
        self.centroid = (self.centroid * len(self.members) + p)/(1+len(self.members))
        self.members.append(p)
        self.member_indices.append(j)

    def dist(self, p):
        """
        Returns the distance of point p to the centroid of the clique
        """
        return np.linalg.norm(p-self.centroid)



@profile
def leaders_cluster(points, vocab, K, output_path, tau=None, ref='', is_fast=True, ann_file=None):
    """
    Uses the point.npy, vocab.npy files of a layer (generated using https://github.com/hsajjad/ConceptX/ library) to produce a clustering of <K> clusters for threshold <tau> at <output_path> named clusters-leaders-{K}-{tau}.txt
    If the threshold tau is not provided, it's esitmated
    is_fast uses approximate nearest neighbor for efficiency and generate a '<output_path>/leaders_{ref}.ann' index file
    if the '.ann' index file has been generated, it could be passed to the function to skip regeneration
    """
    cliques = []

    if not is_fast:
        assert tau is not None
        for j, p in enumerate(points):
            if (j - 1) %1000 == 0: print(j - 1, np.round(len(cliques)/j*100,2))
            found = False
            for c in cliques:
                if c.dist(p) < tau:
                    c.add(p,j)
                    found = True
                    break
            if not found:
                cliques.append(Clique(p, j))

    else:
        t = AnnoyIndex(points.shape[1], 'euclidean')
        if not ann_file:
            for i, p in enumerate(points):
                t.add_item(i,p)
            t.build(1000)
            t.save(f'{output_path}/leaders_{ref}.ann')
        else:
            t.load(ann_file)

        
        # estimate tau?
        if tau is None:
            m = np.random.choice(range(points.shape[0]), replace=False, size=(1000))
            dists_tau = [t.get_nns_by_item(i, 2, include_distances=True)[1] for i in m]
            tau = statistics.median([d[1] for d in dists_tau])

        used_indices = [0] * points.shape[0]
        for i, p in enumerate(points):
            if used_indices[i] != 0:
                continue
            ul = 100
            found = False
            while not found:
                neighbours, dists = t.get_nns_by_item(i, ul, include_distances=True)
                if dists[-1] < tau:
                    ul *= 2
                else:
                    for j in range(len(neighbours)):
                        if dists[j] > tau:
                            ul = j
                            found = True
                            break

            cliques.append(Clique(points[neighbours[0], :], neighbours[0]))
            used_indices[neighbours[0]] = 1
            for n in neighbours[1:ul]:
                if used_indices[n] == 1:
                    continue
                cliques[-1].add(points[n, :],n)
                used_indices[n] = 1
            if len(cliques) % 100 == 0:
                print(f'Cliques {len(cliques)} -- Points {sum(used_indices)}/{points.shape[0]}')

    centroids = [c.centroid for c in cliques]

    clustering = AgglomerativeClustering(n_clusters=K,compute_distances=True).fit(centroids)

    word_clusters = defaultdict(list)
    for i,label in enumerate(clustering.labels_):
        word_clusters[label].extend([vocab[u] for u in cliques[i].member_indices])

    out = ""
    for key, words in word_clusters.items():
        for word in words:
            out += f"{word}|||{key}\n"

    with open(f'{output_path}/clusters-leaders-{K}-{tau}{ref}.txt','w') as of:
        of.write(out)


    return out, tau



parser = argparse.ArgumentParser()
parser.add_argument("--vocab-file","-v", help="input vocab file with complete path")
parser.add_argument("--point-file","-p", help="output point file with complete path")
parser.add_argument("--output-path","-o", help="output path clustering model and result files")
parser.add_argument("--cluster","-k", help="cluster number")
parser.add_argument("--count","-c", help="point count ratio", default=-1)
parser.add_argument("--tau","-t", help="Leaders threshold")
parser.add_argument("--fast", action='store_true')
parser.add_argument("--ann", '-a', help="ann file to load")

args2 = parser.parse_args()
vocab_file = args2.vocab_file
point_file = args2.point_file
output_path = args2.output_path
is_fast = args2.fast
ann_file = args2.ann

point_count_ratio = float(args2.count)
K = int(args2.cluster)

vocab = np.load(vocab_file) 
original_count = len(vocab)
useable_count =	int(point_count_ratio*original_count) if point_count_ratio != -1 else -1
vocab = np.load(vocab_file)[:useable_count]

points = np.load(point_file)[:useable_count, :]

tau = float(args2.tau) if args2.tau is not None else None
K = int(args2.cluster)
ref = "-"+str(point_count_ratio) if point_count_ratio>0 else ""

start_time = time.time()
leaders_cluster(points, vocab, K, output_path, tau, ref, is_fast=is_fast, ann_file=ann_file)
end_time = time.time()

print(f"Runtime: {end_time - start_time}")

