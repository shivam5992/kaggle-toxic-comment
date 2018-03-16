def parse_it(inp):
	tree = {}
	for node in inp:
		tree[node] = {}
		for each in inp[node]:
			tree[node][each[0]] = {}

	visited = {}
	for k,v in tree.iteritems():
		for v1 in v:
			if v1 in tree and v1 not in visited:
				visited[v1] = 1
				tree[k][v1] = tree[v1]

	updated = {}	
	for k,v in tree.iteritems():
		if k not in visited:
			updated[k] = v 
	output = str(updated).replace(":","").replace("{","(").replace("}",")")
	return output

inp = {2: [(0, u'amod'), (1, u'compound')], 
	 4: [(2, u'nsubj'), (3, u'aux'), (6, u'dobj')], 
     6: [(5, u'det'), (10, u'nmod'), (13, u'nmod')], 
     8: [(9, u'case')], 
     10: [(7, u'case'), (8, u'nmod:poss')], 
     13: [(11, u'case'), (12, u'det'), (16, u'nmod'), (21, u'nmod')], 
     16: [(14, u'case'), (15, u'nummod')], 
     21: [(17, u'case'), (18, u'det'), (19, u'compound'), (20, u'compound')]}

inp = {1: [(0, u'nsubj'), (3, u'dobj'), (6, u'punct')], 3: [(2, u'det'), (5, u'nmod')], 5: [(4, u'case')]}

pattern = {'pos':'VB','reln':"", 'subtree':[{'pos': '', 'reln':'nsubj','subtree':[]}, {'pos':'', 'reln':'dobj', 'subtree':[{'pos':'','reln':'nmod','subtree':[]}]}]}

def get_pattern(k_pos, k_reln, l_pos):
	found = []
	for k,v in inp.iteritems():
		if k_pos == pattern['pos'] and k_reln == pattern['reln']:
			found.append(k)
			for each in v:
				if each[1] == 'nsubj':
					found.append(each[0])
					get_pattern(each[1], pattern[''])

				if each[1] == 'dobj':
					found.append(each[0])
			break




# pattern = {'pos':'VB','reln':"", 'subtree':[{'pos': '', 'reln':'nsubj','subtree':[]}, {'pos':'', 'reln':'dobj', 'subtree':[{'pos':'','reln':'nmod','subtree':[]}]}]}

# Allseas signed a contract with Statoil.
# tree1 = "(1-signed-VBD-None (0-Allseas-NNP-nsubj (), 3-contract-NN-dobj (2-a-DT-det (), 5-Statoil-NNP-nmod (4-with-IN-case ())), 6-.-.-punct ()))"
# tree2 = "(token1 (token2 (), token3 (token4 (), token5 (token6 ())), token7 ()))"
# updated_tree = {'root' : "", 'subtree' : []}

