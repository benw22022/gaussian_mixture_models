"""
Script to convert csv files to root
"""

import os
import glob
import uproot
import pandas as pd
import numpy as np
import copy

def get_params(filepath, is_SM=False):
    
    if is_SM:
        return [1]
    
    csv_name = os.path.basename(filepath)
    params = []
    for elem in csv_name.split("_"):
        try:
            params.append(float(elem))
        except ValueError:
            pass
    return params


        

def main(input_data, output_file, is_sm):
    
    filepaths = glob.glob(input_data)
    
    if not isinstance(filepaths, list):
        filepaths = [filepaths]
    
    features = ['weight',
            'eTmiss',
            'lep1_pT',
            'lep2_pT',
            'dilep_pT',
            'jet1_pT',
            'jet2_pT',
            'dijet_pT',
            'HT',
            'mjj',
            'mll',
            'mlljj',
            'ml1jj',
            'ml2jj',
            'delta_rap_jj',
            'delta_rap_ll',
            'rap1',
            'rap2',
            'delta_phi_jj',
            'delta_phi_ll',
            'delta_R_jj',
            'delta_R_ll',
            'delta_R_lljj',
            'njets']
    
    with uproot.recreate(output_file) as outfile:
        
        df = pd.read_csv(filepaths[0])
        branch_dict = {}
        
        for f in features:
            branch_dict[f] = df[f].to_numpy()
        
        if not is_sm:
            mw2, mn, _, _ = get_params(filepaths[0])
            branch_dict['external_params'] = np.full((len(df), 2), np.array([mw2, mn]), dtype='float32')
        else:
            branch_dict['external_params'] = np.full((len(df), 1), np.array([1]), dtype='float32')
        
        
        outfile['tree'] = branch_dict
        
        
        print(f'Processed {filepaths[0]}')
        
        
        for i in range(1, len(filepaths)):
            
            df = pd.read_csv(filepaths[i])
            branch_dict = {f: copy.deepcopy(df[f]) for f in features}
        
            if not is_sm:
                mw2, mn, _, _ = get_params(filepaths[0])
                branch_dict['external_params'] = np.full((len(df), 2), np.array([mw2, mn]), dtype='float32')
            else:
                branch_dict['external_params'] = np.full((len(df), 1), np.array([1]), dtype='float32')
            
            outfile['tree'].extend(branch_dict)
            
            print(f'Processed {filepaths[i]}')
    
    
        data = uproot.concatenate(output_file)
        print(data['external_params'])
        return 
        
    
if __name__ == "__main__":
    main("../data/*.csv", "../data/SM_ssWW.root", is_sm=True)
    

