
# convert tree structure to table info

# input : lightgbm model text file
# output : table containing the belows
#           'split_index', 'split_feature', 'split_gain',
#           'threshold', 'decision_type', 'default_value', 'internal_value', 'internal_count',
#           'leaf_index', 'leaf_value', 'leaf_count',
#           'depth', 'node_parent', 'tree_idx'


import pandas as pd
import lightgbm as lgb

extract_columns = ['split_index', 'split_feature', 'split_gain',
           'threshold', 'decision_type', 'default_value', 'internal_value', 'internal_count',
           'leaf_index', 'leaf_value', 'leaf_count']
all_columns = extract_columns + ['depth', 'node_parent', 'tree_idx']


def parse_tree(model_file):
    bst = lgb.Booster(model_file=model_file)

    model_json = bst.dump_model()

    tree_info = model_json['tree_info']
    num_trees = len(tree_info)

    rows = []
    for tree_id in range(num_trees):
        rows.extend(single_tree_parse(tree_id, tree_info[tree_id])['node_contents'])
    df = pd.DataFrame(rows, columns = all_columns)

    return df



def single_tree_parse(tree_index, single_tree_info):
    return pre_order_traversal(tree_index, None, single_tree_info['tree_structure'], 0, None)


def parse_node_property(node, depth, node_parent, tree_idx):

    columns = ['split_index', 'split_feature', 'split_gain',
               'threshold', 'decision_type', 'default_value', 'internal_value', 'internal_count',
               'leaf_index', 'leaf_value', 'leaf_count']

    extracted_values = []
    for column in columns:
        if column in node:
            extracted_values.append(node[column])
        else:
            extracted_values.append(None)

    extracted_values.extend([depth, node_parent, tree_idx])

    return extracted_values

def pre_order_traversal(tree_index, context=None, tree_node_leaf=None, current_depth=0, parent_index=None):

    if context is None:
        # initialize default context

        context = {
            'node_contents': []
        }

        # start tree traversal
        pre_order_traversal(tree_index, context, tree_node_leaf, current_depth, parent_index)

    else:
        if 'split_index' in tree_node_leaf: # not leaf node

            context['node_contents'].append(parse_node_property(tree_node_leaf, current_depth, parent_index, tree_index))

            # go to left child
            pre_order_traversal(tree_index,
                                context,
                                tree_node_leaf['left_child'],
                                current_depth = current_depth + 1,
                                parent_index = tree_node_leaf['split_index']
                                )

            # go to right child
            pre_order_traversal(tree_index,
                                context,
                                tree_node_leaf['right_child'],
                                current_depth = current_depth + 1,
                                parent_index = tree_node_leaf['split_index']
                                )

        elif 'leaf_index' in tree_node_leaf: # leaf node
            context['node_contents'].append(parse_node_property(tree_node_leaf, current_depth, parent_index, tree_index))

    return context


if __name__ == '__main__':
    df = parse_tree('model.txt')

    print(df[df['split_gain'] >= 40][['split_index', 'split_gain', 'tree_idx']].head())

    # output will be
    #     split_index  split_gain  tree_idx
    # 0           0.0   82.983101         0
    # 1           1.0   56.344898         0
    # 19          0.0   68.228699         1
    # 24          2.0   43.838001         1
    # 38          0.0   56.180099         2