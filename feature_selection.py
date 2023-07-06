import pandas as pd
import statsmodels.api as sm

class FeatureSelection(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(data, target, significance_level=0.05): # 9 pts
        '''
        Implement forward selection using the steps provided in the notebook.
        You can use sm.OLS for your regression model.
        Do not forget to add a bias to your regression model. A function that may help you is the 'sm.add_constants' function.
        
        Args:
            data: data frame that contains the feature matrix
            target: target feature to search to generate significant features
            significance_level: the probability of the event occuring by chance
        Return:
            forward_list: list containing significant features (in order of selection)

        '''

        keep_feats = []
        all_feats = data.columns.tolist()

        min_pval = 0
        while min_pval <= significance_level: # until min pval too big

            while len(all_feats) > 0:

                feat_dict = {} # feat:pvalue
                
                for feature in all_feats:

                    testlist = keep_feats
                    testlist.append(feature)
                    # create model:
                    subset = pd.DataFrame(data[testlist]) # subset columns
                    new_data = sm.add_constant(subset) # add constant
                    regress = sm.OLS(target, new_data).fit() # model

                    feat_dict[feature] = regress.pvalues[feature]

                    testlist.remove(feature)

                # determine which feature's pvalue was the lowest; add to keep_feats
                keep = min(feat_dict.keys(), key=(lambda k: feat_dict[k]))
                min_pval = feat_dict[keep]

                if min_pval < significance_level:
                    keep_feats.append(keep)

                # remove from all_feats
                all_feats.remove(keep)

        #print(keep_feats)

        return keep_feats 

        raise NotImplementedError

    @staticmethod
    def backward_elimination(data, target, significance_level = 0.05): # 9 pts
        '''
        Implement backward selection using the steps provided in the notebook.
        You can use sm.OLS for your regression model.
        Do not forget to add a bias to your regression model. A function that may help you is the 'sm.add_constants' function.

        Args:
            data: data frame that contains the feature matrix
            target: target feature to search to generate significant features
            significance_level: the probability of the event occuring by chance
        Return:
            backward_list: list containing significant features
            removed_features = list containing removed features (in order of removal)
        '''

        all_feats = data.columns.tolist()
        discard = []

        max_pval = significance_level + 1

        while max_pval > significance_level:

            # create model:
            subset = pd.DataFrame(data[all_feats]) # subset columns
            new_data = sm.add_constant(subset) # add constant
            regress = sm.OLS(target, new_data).fit() # model

            # get feature with the highest pvalue
            max_pval = max(regress.pvalues)
            feat = max(regress.pvalues.keys(), key=(lambda k: regress.pvalues[k]))

            if max_pval > significance_level:

                all_feats.remove(feat)
                discard.append(feat)

            elif max_pval <= significance_level:

                break

            #print(discard)

        return all_feats, discard

        raise NotImplementedError






















