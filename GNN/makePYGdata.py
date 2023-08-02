import torch
from torch_geometric.data import Data, DataLoader, InMemoryDataset, download_url
from collections import defaultdict
from tqdm import tqdm
import csv
import datasets
from datasets import load_dataset, Dataset

import benepar, spacy
nlp = spacy.load('en_core_web_md')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

# https://tedboy.github.io/nlps/generated/generated/nltk.ParentedTree.html
import nltk
from nltk import Tree


################################################################################################

class PTC_Trees_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        
        

#     @property
#     def raw_file_names(self):
#         return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['ptc_trees.pt']

#     def download(self):
#         # Download to `self.raw_dir`.
#         download_url(url, self.raw_dir)
#         ...

    def process(self):
        
        # print(self)
        
        
        def parseText(text):
            """
            input:  sentence - EX: 'He runs slowly'
            output: nltk parse tree and a parse string - EX: '(S (NP (PRP He)) (VP (VBZ runs) (ADVP (RB slowly))))'
            """
            doc = nlp(text)
            sent = list(doc.sents)[0]
            parse_tree = Tree.fromstring('(' + sent._.parse_string + ')')
            parse_string = sent._.parse_string
            return (parse_tree, parse_string)

        def tagsToNodeIndexes(parse_string):
            """
            input: sent._.parse_string - EX: '(S (NP (PRP He)) (VP (VBZ runs) (ADVP (RB slowly))))'
            outputs: 0: new parse string with node labels converted to indexes - EX: '(0 (1 (2 He)) (3 (4 runs) (5 (6 slowly))))'
                     1: dictionary of node labels by index
            """
            nodes_dict = defaultdict(int)
            new_string = ""
            idx = 0

            for item in parse_string.split(" "):
                if item.startswith('('):
                    nodes_dict[idx]=item[1:]
                    new_string+='('+str(idx)+" "
                    idx += 1
                else:
                    new_string+=item+" "
            return (new_string.strip(), nodes_dict)


        def convertToEdgeList(tree):
            """
            input: NLTK Tree
            output: list of tuples of edges of the form [[source_node, destination_node],...]
            """
            edge_list = []

            for production in tree.productions():
                for elem in production.rhs():
                    if type(elem) is not str and production.lhs().symbol() != "":
                        # [source, destination]
                        edge_list.append([int(str(production.lhs())), int(str(elem))])
                        # for undirected graph, also include [destination, source]:
                        edge_list.append([int(str(elem)), int(str(production.lhs()))])
            return edge_list


        def makeDatum(edges,nodes_ids,class_name,split):
            edge_index = torch.tensor(edges, dtype=torch.long)
            x = torch.tensor(nodes_ids, dtype=torch.float)
            y = torch.tensor([class_name], dtype=torch.long)
            split = torch.tensor([split], dtype=torch.uint8) # 8-bit integer (unsigned)
            datum = Data(x=x, y=y, edge_index=edge_index.t().contiguous(), split=split)
            
            return datum



        tags_to_ids = defaultdict(str)
        ids_to_tags = defaultdict(int)

        with open('pennTreeBankTags.txt','r') as _f:
            lines = _f.readlines()
            for idx,line in enumerate(lines):
                ids_to_tags[idx]=line.strip()
                tags_to_ids[line.strip()]=idx

        # Load the PTC Corpus and process
        dataset = load_dataset("Kyleiwaniec/PTC_Corpus",
                               use_auth_token='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')

        
        def update_labels(example):
            example['labels'] = example['labels'][0] if len(example['labels']) else 18
            return example

        classification = 'multi'
        if classification == 'multi':
            # For multiclass classification use the technique classification as labels
            dataset = dataset.rename_column("labels", "binary_labels")
            dataset = dataset.rename_column("technique_classification", "labels")
            dataset = dataset.map(update_labels, num_proc=4)


        # sample = dataset['train'].select(range(0,20))


        data_list = []
        splits = ['train','validation','test']
        
                
        unknown_tags = []
        for split in splits:
            d = tqdm(dataset[split])
            for sentence in d:
                d.set_description("processing sentence:", refresh=True)
                """
                dataset:
                DatasetDict({
                    validation: Dataset({
                        features: ['article_id', 'text', 'technique_classification', 'offsets', 'labels'],
                        num_rows: 2067
                    })
                    test: Dataset({
                        features: ['article_id', 'text', 'technique_classification', 'offsets', 'labels'],
                        num_rows: 4083
                    })
                    train: Dataset({
                        features: ['article_id', 'text', 'technique_classification', 'offsets', 'labels'],
                        num_rows: 14434
                    })
                })
                """
                text = sentence['text']
                class_name = int(sentence['labels'])

                tree, text_parse = parseText(text)
                new_parse, nodes_dict = tagsToNodeIndexes(text_parse)

                t = Tree.fromstring(new_parse)
                edges = convertToEdgeList(t)
                
                
                # unknown tag -> X or 94
                # nodes = [v for k,v in nodes_dict.items()]
                # nodes_ids = [[tags_to_ids[n]/100] if n in tags_to_ids.keys() else [0.83] for n in nodes]
                nodes_ids = [[tags_to_ids[v]/100] if v in tags_to_ids.keys() else [0.94] for k,v in nodes_dict.items()]
                
                ###################################################
                """
                in case we want to find additioinal unknown tags
                replace the above list comprehension with this loop
                """
                # nodes_ids = []
                # for n in nodes:
                #     if n not in tags_to_ids.keys():
                #         unknown_tags.append([n,text])
                #         nodes_ids.append([0.94])
                #     else:
                #         nodes_ids.append([tags_to_ids[n]/100])
                ###################################################
                
                
                datum = makeDatum(edges,nodes_ids,class_name,splits.index(split))
                data_list.append(datum)
        
        ###################################################
        """
        write unknown tags to file
        """
        # with open('unknown_tags_val.csv', 'w') as f:
        #     write = csv.writer(f)
        #     write.writerow(['TAG','TEXT']) #header
        #     write.writerows(unknown_tags)
        ###################################################
            
################################################################################################        

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
################################################################################################       
        
        