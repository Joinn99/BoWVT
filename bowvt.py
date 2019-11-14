import os
import argparse
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import utils


class VocabularyTree():
    def __init__(self, arg):
        self.file_dir = './'

        self.tree = {}                      # Tree Structure
        self.nodes = {}                     # Tree Node Feature
        self.param = {}                     # BOW Parameters

        self.textf = None                   # Text Frequency
        # self.invert = {}                  # Invert File Index

        self.args = arg                    # Arguments
        self.node_index = 0

    def _extract_features(self, file_dir, file_list):
        if self.args.pre_allocate:
            print('Calculating needed memory size...')
            # feature_num = calculate_memory(file_dir, file_list, self.args)
            feature_num = 1168863 # Only for this dataset
            print('\rAll feature numbers: {:8d} Memory: {:8d} KB'.format(
                feature_num, int(feature_num / 2)))
            features = np.empty((feature_num, 128))
        else:
            features = np.empty((0, 128))           # All Image Features
        feature_index = []                          # Feature Index

        for ind, file in enumerate(file_list):
            descs, _ = utils.sift(file_dir + os.sep + file, self.args.scale)
            if self.args.pre_allocate:
                start = len(feature_index)
                features[start:start+descs.shape[0]] = descs
            else:
                features = np.vstack((features, descs))
            feature_index.extend((ind,) * descs.shape[0])
            print('Image {:4d}\t---\tFeature Number {:5d}'.format(
                ind, descs.shape[0]), flush=True, end='\r')

        return features, feature_index

    def _gen_tree(self, node, depth, feature_ids, features):
        self.tree[node] = []
        print(
            'Vocabulary Tree: Node:{:d} --- Depth:{:d}'.format(node, depth), flush=True, end='\r')
        kmeans = MiniBatchKMeans(n_clusters=self.args.branch)
        if len(feature_ids) > 2 * self.args.branch and depth < self.args.max_depth:
            kmeans.fit(features[feature_ids])
            child = [[] for ind in range(self.args.branch)]
            for ind, fid in enumerate(feature_ids):
                child[kmeans.labels_[ind]].append(fid)
            for ind in range(self.args.branch):
                self.node_index += 1
                self.nodes[self.node_index] = kmeans.cluster_centers_[ind]
                self.tree[node].append(self.node_index)
                self._gen_tree(self.node_index, depth +
                               1, child[ind], features)
        else:
            # self.invert[node] = {}
            pass

    def _lookup(self, desc, cur_node):
        child_desc = np.array([self.nodes[ind] for ind in self.tree[cur_node]])
        dist = np.linalg.norm(child_desc - desc, axis=1)
        next_node = np.argmin(dist)
        if self.tree[self.tree[cur_node][next_node]] == []:
            return self.tree[cur_node][next_node]
        else:
            return self._lookup(desc, self.tree[cur_node][next_node])

    def _tfidf(self, features, feature_index):
        for ind, desc_ind in enumerate(feature_index):
            leaf_id = self._lookup(features[ind], 0)
            self.textf[desc_ind][leaf_id] += 1
            # if desc_ind in self.invert[leaf_id]:
            #     self.invert[leaf_id][desc_ind] += 1
            # else:
            #     self.invert[leaf_id][desc_ind] = 1
            if ind % 1000:
                print(
                    'Inverted file indexing --- {:9d} features'.format(ind), flush=True, end='\r')
        self.update(0, -1, self.textf)
        self.param['Node_N'] = np.count_nonzero(self.textf, axis=0)
        self.param['Node_W'] = np.log1p(
            len(feature_index) / self.param['Node_N'])

    def _calculate_score(self, n_src, norm_ord):
        v_data = self.textf * self.param['Node_W']
        v_query = n_src * self.param['Node_W']
        v_data = v_data / \
            np.linalg.norm(v_data, ord=norm_ord, axis=1, keepdims=True)
        v_query = v_query / \
            np.linalg.norm(v_query, ord=norm_ord, axis=1, keepdims=True)
        # v_mask = (v_data != 0) * (v_query != 0)
        # scores = 2 + utils.abs_norm((v_query - v_data) * v_mask, norm_ord) - \
        #     utils.abs_norm(v_query * v_mask, norm_ord) - \
        #     utils.abs_norm(v_query * v_mask, norm_ord)
        scores = utils.abs_norm(v_query - v_data, norm_ord)
        return scores

    # Need Edit

    def _spatial_check(self, src_kpt, src_ind, match_list):
        scores = []
        for ind_p, path in enumerate(match_list):
            data_desc, data_kpt = utils.sift('static/' + path, self.args.scale)
            data_ind = np.zeros((1, data_desc.shape[0]))
            for ind, desc in enumerate(data_desc):
                data_ind[0, ind] = self._lookup(desc, 0)
            scores.append(utils.ransac(src_kpt, data_kpt,
                                       np.nonzero(src_ind.T == data_ind)))
            print('Image {:2d} --- {:4d} keypoints matches'.format(
                ind_p + 1, scores[ind_p]), flush=True, end='\r')
        return np.argsort(-1 * np.array(scores))

    def _gen_feature(self, image_path):
        descs, kpts = utils.sift(image_path, self.args.scale)
        n_src = np.zeros((1, len(self.nodes)))
        kpts_ind = np.zeros((1, descs.shape[0]))
        for ind, desc in enumerate(descs):
            leaf_id = self._lookup(desc, 0)
            kpts_ind[0, ind] = leaf_id
            n_src[0, leaf_id] += 1
        self.update(0, -1, n_src)
        return n_src, kpts, kpts_ind

    def _get_image(self, image_id):
        file_list = sorted([img for img in os.listdir(
            self.file_dir) if img.endswith('.jpg') or img.endswith('.png')])
        img_list = []
        for imid in image_id:
            img_list.append(str(self.file_dir + os.sep +
                                file_list[imid]).replace('static/', ''))
        return img_list
        # draw_result(img_list)

    def construct(self, file_dir='./'):
        self.file_dir = file_dir
        file_list = sorted([img for img in os.listdir(
            file_dir) if img.endswith('.jpg') or img.endswith('.png')])
        print('Step1 | Training Images Number: {:7d}'.format(len(file_list)))

        features, feature_index = self._extract_features(file_dir, file_list)
        print(
            '-' * 55 + '\nStep2 | Total Feature Number: {:9d}'.format(len(feature_index)))

        self.nodes[0] = np.mean(features, axis=0)
        self._gen_tree(
            0, 1, [ind for ind in range(features.shape[0])], features)
        print(
            '-' * 55 + '\nStep3 | Total Node Number: {:4d}'.format(self.node_index+1))

        self.textf = np.zeros((len(file_list), len(self.nodes)))
        self.param['Node_N'] = np.zeros(len(self.nodes), dtype=int)
        self.param['Node_W'] = np.zeros(len(self.nodes), dtype=float)
        self._tfidf(features, feature_index)
        print('-' * 55 + '\nStep4 | TF-IDF Finished')

        pkl_name = 'bow-{:.2f}-{:d}-{:d}.pkl'.format(
            self.args.scale, self.args.max_depth, self.args.branch)
        joblib.dump((self.tree, self.nodes, self.param, self.textf, self.file_dir),
                    'model/' + pkl_name, compress=3)

    def update(self, cur_node, father_node, textf):
        if self.tree[cur_node] != []:
            for child in self.tree[cur_node]:
                self.update(child, cur_node, textf)
        if father_node != -1:
            textf[:, father_node] += textf[:, cur_node]

    def load(self, file_path):
        self.tree, self.nodes, self.param, self.textf, self.file_dir = joblib.load(
            file_path)
        print('-' * 55 + '\n-VT-  | Load from [\'{:s}\']\n\tVocabulary tree: {:d} nodes'.format(
            file_path, len(self.nodes)))
        print('\tImage database: {:d} images'.format(
            self.textf.shape[0]))

    def search(self, image_path, norm_ord=2):
        print('-' * 55 + '\nSearching image \'[{:s}]\''.format(image_path))
        n_src, kpts, kpts_ind = self._gen_feature(image_path)
        scores = self._calculate_score(n_src, norm_ord)
        rank_id = np.argsort(scores.reshape(-1))[:36]
        print('Performing spatial check...')
        # return self._get_image(rank_id[:12])
        final = rank_id[self._spatial_check(
            kpts, kpts_ind, self._get_image(rank_id))]
        print('-' * 55 + '\nSearch complete.')
        return self._get_image(final[:12])

    def optimize(self, target, result):
        n_src, kpts, kpts_ind = self._gen_feature(target)
        delta_w = np.zeros((1, len(self.nodes)))
        for key, value in result.items():
            m_src, _, _ = self._gen_feature('static/' + key)
            delta_w -= np.abs(m_src - n_src) * int(value)
        self.param['Node_W'] *= (1 + 0.05 * (delta_w -
                                             np.mean(delta_w)) / np.std(delta_w))[0]
        scores = self._calculate_score(n_src, norm_ord=2)
        rank_id = np.argsort(scores.reshape(-1))[:36]
        # return self._get_image(rank_id[:12])
        final = rank_id[self._spatial_check(
            kpts, kpts_ind, self._get_image(rank_id))]
        print('-' * 55 + '\nSearch complete.')
        return self._get_image(final[:12])


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-d", "--max_depth",
                        help="Max depth of vocab tree", default=4, type=int)
    PARSER.add_argument(
        "-b", "--branch", help="Branches of each node", default=12, type=int)
    PARSER.add_argument(
        "-s", "--scale", help="Scale image", default=0.5, type=float)
    PARSER.add_argument(
        "-p", "--pre_allocate", help="Pre-Allocate storage for features", default=True, type=bool)

    ARGS = PARSER.parse_args()

    VT = VocabularyTree(ARGS)
    VT.construct(file_dir='static/data')
    # VT.load('model/bow-0.50-4-12.pkl')
    # VT.search('data/magdalen_001152.jpg', norm_ord=2)
