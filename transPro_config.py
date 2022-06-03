import ml_collections
import pickle

latent_space_dim_placeholder = ml_collections.FieldReference(128)
all_cell_file_dir = '/raid/home/yoyowu/PertPro/perturbed_proteomics/data/ccle_tcga_ad_cells.p'
all_cells = list(pickle.load(open(all_cell_file_dir, 'rb')))
CONFIG = ml_collections.ConfigDict({
    'data': {
        'data_filter': {
            "time": "24H",
            "pert_id": ['BRD-U41416256', 'BRD-U60236422','BRD-U01690642','BRD-U08759356','BRD-U25771771', 'BRD-U33728988', 'BRD-U37049823',
            'BRD-U44618005', 'BRD-U44700465','BRD-U51951544', 'BRD-U66370498','BRD-U68942961', 'BRD-U73238814',
            'BRD-U82589721','BRD-U86922168','BRD-U97083655'],
            "pert_type": ["trt_cp"],
            "cell_id": all_cells,
            "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]
            }
    },
    'model': {
        'cell_encoder':{
            'cell_dim': 978,
            'linear_encoder_flag': True,
            'cell_hidden_dim': 512,
            'latent_space_dim': latent_space_dim_placeholder
        },
        'pros_encoder':{
            'input_dim': 512,
            'output_dim':latent_space_dim_placeholder,
            'hidden_dim': latent_space_dim_placeholder,
            'dropout': 0.3
        },
        'pros_decoder':{
            'input_dim': 2*latent_space_dim_placeholder,
            'output_dim': 512,
            'hidden_dim': 256,
            'dropout': 0.3
         },
         'transmitter':{
            'input_dim': 2*latent_space_dim_placeholder,
            'output_dim': 256,
            'hidden_dim': 128,
            'dropout': 0.3
        },
        'trans_decoder':{
            'input_dim': 2*latent_space_dim_placeholder,
            'output_dim': 978,
            'hidden_dim': 256,
            'dropout': 0.3
        },
        'drug_network' :{
            'num_layer':5,
            'emb_dim' : 300,
            'JK' : 'last',
            'gnn_type' :'gin'
        },

        'diff_generator':{
            'input_dim': 300,
            'output_dim': latent_space_dim_placeholder,
            'hidden_dim':256,
            'dropout': 0.3
        },
        'drug_cell_attn':{
            'hidden_dim': latent_space_dim_placeholder
        },
        'perturbed_trans':{
            'loss_type': 'point_wise_mse'
        },
        'perturbed_pros':{
            'loss_type': 'weighted_point_wise_mse_adjN' #weighted_point_wise_mse'
        },      
        'trans_autoencoder':{
                'loss_type': 'point_wise_mse'
            },
        'pros_autoencoder':{
            'loss_type': 'point_wise_mse'
        },
        'transmitter_only':{
            'loss_type': 'point_wise_mse'
        },
        'trans_transmitter_pros':{
            'loss_type': 'point_wise_mse' 
        },
    }
})

def get_config(config_string):
    return CONFIG[config_string]



