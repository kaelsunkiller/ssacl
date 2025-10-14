def init_anatomy_tree():
        anatomical_tree = {
            "chest": {
                "lungs": {
                    "right lung": {
                        "right upper lung zone": {},
                        "right mid lung zone": {},
                        "right lower lung zone": {},
                        "right hilar structures": {},
                        "right apical zone": {},
                        "right costophrenic angle": {},
                        "right hemidiaphragm": {}
                    },
                    "left lung": {
                        "left upper lung zone": {},
                        "left mid lung zone": {},
                        "left lower lung zone": {},
                        "left hilar structures": {},
                        "left apical zone": {},
                        "left costophrenic angle": {},
                        "left hemidiaphragm": {}
                    }
                },
                "mediastinum": {
                    "trachea": {},
                    "upper mediastinum": {},
                    "aortic arch": {},
                    "superior vena cava": {},
                    "cardiac silhouette": {
                        "right atrium": {},
                        "cavoatrial junction": {}
                    },
                    "carina": {}
                },
                "skeletal structures": {
                    "spine": {},
                    "right clavicle": {},
                    "left clavicle": {}
                },
                "abdomen": {}
            }
        }
        def _dict2list(tree):
            currentroots = list(tree.keys())
            currenttrees = []
            currentnodes = list(tree.keys())
            currentleafnodes = []
            for k, v in tree.items():
                if v:
                    subroots, subtrees, subnodes, subleafnodes = _dict2list(v)
                    currentnodes.extend(subnodes)
                    currenttrees.append([k, subroots])
                    currenttrees.extend(subtrees)
                    currentleafnodes.extend(subleafnodes)
                else:
                    currentleafnodes.append(k)
            return currentroots, currenttrees, currentnodes, currentleafnodes
        root, trees, all_nodes, all_leafnodes = _dict2list(anatomical_tree)
        treeids = [[all_nodes.index(k), [all_nodes.index(sv) for sv in v]] for k, v in trees]
        leafids = [all_nodes.index(k) for k in all_leafnodes]
        return root, trees, treeids, all_nodes, all_leafnodes, leafids
    
def roots2leafs(roots):
    _r2f = {
            "lungs": [
                    "right upper lung zone",
                    "right mid lung zone",
                    "right lower lung zone",
                    "right hilar structures",
                    "right apical zone",
                    "right costophrenic angle",
                    "right hemidiaphragm",
                    "left upper lung zone",
                    "left mid lung zone",
                    "left lower lung zone",
                    "left hilar structures",
                    "left apical zone",
                    "left costophrenic angle",
                    "left hemidiaphragm",
            ],
            "mediastinum": [
                "trachea",
                "upper mediastinum",
                "aortic arch",
                "superior vena cava",
                "right atrium",
                "cavoatrial junction",
                "carina",
            ],
            "skeletal structures": [
                "spine",
                "right clavicle",
                "left clavicle",
            ],
        }
    outputs = []
    for r in roots:
        if r == "abdomen":
            outputs.append(r)
        else:
            outputs.extend(_r2f[r])
    return outputs