import pandas as pd
import h5py

class L1000DataBuilder:

    '''
    This class is to help with the preprocessing of the L1000 ph1 and ph2 data.
    '''
    ph1 = None
    ph2 = None

    def __init__(self, ph1, ph2, phase_data_type):
        '''
        :ph1: could be file, dataframe or h5
        :ph2: could be file, dataframe or h5
        :datatype: str: file type of ph1 and ph2: csv_file, dataframe, h5_file
        '''
        self.prepare_ph1_ph2_with_different_formats(ph1, ph2, data_type = phase_data_type)
    
    def retrive_phase_data(self, columns = None, phase_num = 1):
        '''
        Retrive the data based on the phase
        :phase_num: int: 1 or 2
        :columns: list: list of col that wanna be retrived from phase data
        :return; dataframe: 
        '''
        if columns is None:
            columns = self.ph1.columns
        if phase_num == 1:
            return self.ph1[columns]
        else:
            return self.ph2[columns]

    def prepare_ph1_ph2_with_different_formats(self, ph1, ph2, data_type = None):
        '''
        make sure self.ph1 and self.ph2 is dataframe
        :ph1: could be file, dataframe or h5
        :ph2: could be file, dataframe or h5
        :datatype: str: file type of ph1 and ph2: csv_file, dataframe, h5_file
        '''
        if data_type == 'dataframe' and self.ph1 is None:
            self.ph1 = ph1
            self.ph2 = ph2
        elif data_type == 'csv_file' and self.ph1 is None:
            self.ph1 = pd.read_csv(ph1)
            self.ph2 = pd.read_csv(ph2)
        elif data_type == 'h5_file' and self.ph1 is None:
            self.__prepare_ph1_ph2_h5(ph1, ph2)

    def __prepare_ph1_ph2_h5(self, ph1, ph2):
        self.ph1 = self.__transfer_h5_to_df(ph1)
        self.ph2 = self.__transfer_h5_to_df(ph2)
        
    def __transfer_h5_to_df(self, h5_file):
        try:
            h5_data = h5py.File(h5_file, 'r')
            keys = h5_data.keys()
            ph_array = None
            assert len(keys) == 3, "the h5 format is not correct"
            ### get the raw data which should be a matrix
            for key in keys:
                if len(h5_data[key].shape) == 2:
                    ph_array = h5_data[key][:]
                    row_len = ph_array.shape[0]
                    col_len = ph_array.shape[1]
            row_index, col_index = None, None
            ### get the row_index and col_index
            for key in keys:
                if len(h5_data[key].shape) == 1:
                    if len(h5_data[key]) == row_len:
                        row_index = h5_data[key][:]
                    if len(h5_data[key]) == col_len:
                        col_index = h5_data[key][:]
            assert row_len != col_len, "column index length should be different row_index length"
            return pd.DataFrame(ph_array, index = row_index, columns = col_index)
        except:
            raise
        finally:
            h5_data.close()

class ConfidenceDataBuilder:
    '''
    This class is to help with the preprocessing of the L1000 ph1 and ph2 confidence data.
    '''
    ph1_confidence = None
    ph2_confidence = None
    def __init__(self, ph1_confidence, ph2_confidence, datatype):
        '''
        :ph1_confidence: could be file, dataframe
        :ph2_confidence: could be file, dataframe
        :datatype: str: file type of ph1 and ph2: csv_file, dataframe
        '''
        if datatype == 'dataframe':
            self.ph1_confidence = ph1_confidence
            self.ph2_confidence = ph2_confidence
        else:
            self.__prepare_confidence_data_from_file(ph1_confidence, ph2_confidence)

    def __prepare_confidence_data_from_file(self, ph1_confidence, ph2_confidence):
        '''
        This function can build the confidence dataframe from the file name, 
        either Bayesian_GSE92742_with_confidence.csv
        or Bayesian_GSE***_with_confidence.csv
        '''
        self.ph1_confidence = pd.read_csv(ph1_confidence)
        self.ph2_confidence = pd.read_csv(ph2_confidence)

    def retrive_confidence_data(self, columns = None, phase_num = 1):
        '''
        Retrive the confidence data based on the phase
        :phase_num: int: 1 or 2
        :columns: list: list of col that wanna be retrived from confidence data
            "sig_id,pert_id,pert_iname,pert_type,cell_id,pert_dose,pert_dose_unit,pert_idose,
            pert_time,pert_time_unit,pert_itime,distil_id,combinations"
        :return; dataframe: 
        '''
        if columns is None:
            columns = self.ph1_confidence.columns      
        if phase_num == 1:
            return self.ph1_confidence[columns]
        else:
            return self.ph2_confidence[columns]


        


