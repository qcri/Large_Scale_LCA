import argparse

def load_sentences_and_labels(sentence_file, label_file):
    sentences = []
    labels = []

    with open(sentence_file, 'r') as f_sentences, open(label_file, 'r') as f_labels:
        for sentence, label_line in zip(f_sentences, f_labels):
            sentences.append(sentence.strip())
            labels.append(label_line.strip())

    return sentences, labels

def load_clusters(cluster_file):
    clusters = []

    with open(cluster_file, 'r') as f_clusters:
        for line in f_clusters:
            parts = line.strip().split("|||")
            word = parts[0]
            word_frequency = parts[1]
            sentence_index = int(parts[2])
            word_index = int(parts[3])
            cluster_id = parts[4].split()[-1]
            clusters.append((word, word_frequency, sentence_index, word_index, cluster_id))

    return clusters


def filter_label_map(label_map):
    filtered_label_map = {}
    unique_labels = set()
    
    for tag, word_list in label_map.items():
        #unique_words = set(word_list)
        if len(word_list) >= 6:
            filtered_label_map[tag] = set(word_list)
            unique_labels.add(tag)
        else:
            print (tag, word_list)
            
    return filtered_label_map, unique_labels



def create_label_map_2(sentences, labels):

    label_map = {}
    unique_labels = set()

    for sentence_index, label_line in enumerate(labels):
        label_tokens = label_line.split()
        word_tokens = sentences[sentence_index].split()

        for word_index, label in enumerate(label_tokens):

            cluster_words = []

            if (label in label_map):
                cluster_words = label_map[label]

            cluster_words.append(word_tokens[word_index])
            label_map[label] = cluster_words
            unique_labels.add(label)

    return filter_label_map(label_map)
    #return label_map, unique_labels


def extract_words_items(cluster_words):

    word_items = [item[0] for item in cluster_words]
    return word_items

def assign_labels_to_clusters_2(label_map, clusters, threshold):
    
    assigned_clusters = {}
    g_c = group_clusters(clusters)

    for cluster_id, cluster_words in g_c:
        for label_id, label_words in label_map.items():
            word_items = extract_words_items(cluster_words)
            
            x = [value for value in word_items if value in label_words]

            match = (len(x)/len(word_items))

            if (match >= threshold and len(set(word_items)) > 5): #and len(label_words) > 5):
                
                assigned_clusters[cluster_id] = label_id

    return assigned_clusters, len(assigned_clusters) 


def create_label_map(sentences, labels):
    label_map = {}
    unique_labels = set()

    for sentence_index, label_line in enumerate(labels):
        label_tokens = label_line.split()
        word_tokens = sentences[sentence_index].split()

        for word_index, label in enumerate(label_tokens):
            label_map[(word_index, word_tokens[word_index])] = label
            unique_labels.add(label)

    return label_map, unique_labels

def assign_labels_to_clusters(label_map, clusters, threshold):
    assigned_cluster_count = 0
    assigned_clusters = {}

    
    for cluster_id, cluster_words in group_clusters(clusters):

        cluster_label_counts = {}

        for word, _, sentence_index, word_index, _ in cluster_words:

            if (word_index, word) in label_map:
                
                label = label_map[(word_index, word)]
                #print (sentence_index, word_index, word, label)

                if label in cluster_label_counts:
                    cluster_label_counts[label] += 1
                else:
                    cluster_label_counts[label] = 1
            # else:
            #     print (sentence_index, word_index, word, "Not found")


        # print (cluster_label_counts)
        # input()

        word_items = extract_words_items(cluster_words)
        total_words = sum(cluster_label_counts.values())
        if total_words > 0 and len(set(word_items)) > 5:
            max_label = max(cluster_label_counts, key=cluster_label_counts.get, default=None)
            if max_label is not None and (cluster_label_counts[max_label] / total_words) >= threshold:
                print(f"Cluster {cluster_id} assigned label: {max_label}")
                assigned_cluster_count += 1
                assigned_clusters [cluster_id] = max_label
            else:
                print(f"Cluster {cluster_id} assigned label: NONE")
                assigned_clusters [cluster_id] = "NONE"
        else:
            print(f"Cluster {cluster_id} assigned label: NONE")
            assigned_clusters [cluster_id] = "NONE"

    return assigned_clusters, assigned_cluster_count 

def group_clusters(clusters):
    cluster_groups = {}

    for cluster in clusters:
        _, _, _, _, cluster_id = cluster
        if cluster_id in cluster_groups:
            cluster_groups[cluster_id].append(cluster)
        else:
            cluster_groups[cluster_id] = [cluster]

    return cluster_groups.items()


def analyze_clusters(dictionary, unique_labels, assigned_cluster_count):
    
    key_count = {}
    for key, value in dictionary.items():
        
        
        if value in key_count:
            key_count[value] = key_count[value] + 1
        else:
            key_count[value] = 1
    
    print (key_count)        
    print ("Unique tags: ", unique_labels)
    print (assigned_cluster_count)
   
    print ("Number of clusters: ", len(dictionary), len(dictionary)/600*100)
    print ("Number of tags covered: ", len(key_count), len(key_count)/len(unique_labels) * 100)
    print ("Number of Unique tags: ", len(unique_labels))

    overall_alignment = ((assigned_cluster_count/600) + (len(key_count)/len(unique_labels)))/2
    print ("Overall Alignment Score: ", overall_alignment)


def main():
    parser = argparse.ArgumentParser(description="Assign labels to clusters based on word labels in sentences.")
    parser.add_argument("--sentence-file", required=True, help="Path to the sentence file")
    parser.add_argument("--label-file", required=True, help="Path to the label file")
    parser.add_argument("--cluster-file", required=True, help="Path to the cluster file")
    parser.add_argument("--threshold", required=True, help="Alignment threshold")
    parser.add_argument("--method", required=True, help="M1 or M2")
 
    args = parser.parse_args()

    sentences, labels = load_sentences_and_labels(args.sentence_file, args.label_file)
    clusters = load_clusters(args.cluster_file)

    if (args.method == "M1"):
        label_map, unique_labels = create_label_map(sentences, labels)
        assigned_clusters, assigned_cluster_count  = assign_labels_to_clusters(label_map, clusters, int(args.threshold)/100)
    elif (args.method == "M2"):
        label_map, unique_labels = create_label_map_2(sentences, labels)
        assigned_clusters, assigned_cluster_count  = assign_labels_to_clusters_2(label_map, clusters, int(args.threshold)/100)


    print(f"Number of clusters successfully assigned a label: {assigned_cluster_count}")

    analyze_clusters(assigned_clusters, unique_labels, assigned_cluster_count)

if __name__ == "__main__":
    main()
