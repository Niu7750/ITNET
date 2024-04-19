import mne
import numpy as np
import random

class GDF_dataset():

    def __init__(self, filename):
        self.filename = filename

    def load_data_events(self):
        raw_gdf = mne.io.read_raw_gdf(self.filename, stim_channel="auto")
        raw_gdf.load_data()
        data = raw_gdf.get_data()
        for i_chan in range(data.shape[0]):
            # first set to nan, than replace nans by nanmean.
            this_chan = data[i_chan]
            # replace the min element of the row with nan"

            data[i_chan] = np.where(
                this_chan == np.min(this_chan), np.nan, this_chan
            )
            #  replance nan with average value of other elements
            mask = np.isnan(data[i_chan])
            chan_mean = np.nanmean(data[i_chan])
            data[i_chan, mask] = chan_mean

        gdf_events = mne.events_from_annotations(raw_gdf)  # "read envent label"
        raw_gdf = mne.io.RawArray(data, raw_gdf.info, verbose="WARNING")
        # remember gdf events
        raw_gdf.info["gdf_events"] = gdf_events
        cnt = raw_gdf
        events, name_to_code = raw_gdf.info["gdf_events"]
        trial_codes = [7, 8, 9, 10]  # 4 classes
        trial_mask = [ev_code in trial_codes for ev_code in events[:, 2]]
        trial_events = events[trial_mask]  # put envens corresponding to 4,5,6,7 into trial_events
        assert len(trial_events) == 288, "Got {:d} markers".format(
            len(trial_events)
        )
        artifact_trial_mask = np.ones(len(trial_events), dtype=np.uint8)
        events = trial_events
        cnt.info["events"] = events
        cnt.info["artifact_trial_mask"] = artifact_trial_mask
        return cnt

def creat_lei_segment(data,lei_index):
    #class 1 left  corresponding to 7 , class 2 right corresponding to 8
    #class 3 foot  corresponding to 9,class 4 tongue corresponding to 10
    #exact trials by classes，each class has 72 trials,return x ,y corresponding to which class
    #lei_index=1-4  to exact trials corresponding to which class
    #data:train_cnt
    lei_code=[7,8,9,10]
    lei1_index=[i for i in range(288) if data.info['events'][i,2]==lei_code[lei_index-1]]
    lei_x=np.zeros((72,22,875),dtype=np.float32)
    lei_y=np.zeros((72,1),dtype=int)
    for i in range(72):
        cue_point=data.info['events'][lei1_index[i],0]
        lei_x[i]=data._data[:,cue_point+126:cue_point+1001]
        lei_y[i]=lei_code[lei_index-1]
    return lei_x


def Separate_source(data, n_test):
    #288 trials in train_cnt have been spilt into training set(288-n_test) and test set(n_test trials)
    n_test_class=int(n_test/4)
    lei_data=np.zeros((4,72,22,875),dtype=np.float32)
    for i in range(4):
        lei_data[i]=creat_lei_segment(data,i+1)

    a = random.sample(range(0,72),n_test_class)

    b=list(range(0,72))
    c=[b[i] for i in range(72) if not b[i] in a]#228
    #exact n_test_class trials from each class , and use remaining trials  as training set
    train_x = np.concatenate((lei_data[0,(c),:,:],lei_data[1,(c),:,:],
                            lei_data[2,(c),:,:],lei_data[3,(c),:,:]),axis=0)
    train_y = np.array([0,1,2,3]).repeat(72-n_test_class)

    test_x=np.concatenate((lei_data[0,(a),:,:],lei_data[1,(a),:,:],
                            lei_data[2,(a),:,:],lei_data[3,(a),:,:]),axis=0)
    test_y = np.array([0,1,2,3]).repeat(n_test_class)
    return train_x,train_y,test_x,test_y

def Separate_target(data, n_test, index):
    #288 trials in train_cnt have been spilt into training set(288-n_test) and validation set(n_test trials)
    n_test_class=int(n_test/4)
    lei_data=np.zeros((4,72,22,875),dtype=np.float32)
    for i in range(4):
        lei_data[i]=creat_lei_segment(data,i+1)

    a = index

    b=list(range(0,72))
    c=[b[i] for i in range(72) if not b[i] in a]#228
    #exact n_test_class trials from each class , and use remaining trials  as training set
    train_x = np.concatenate((lei_data[0,(c),:,:],lei_data[1,(c),:,:],
                            lei_data[2,(c),:,:],lei_data[3,(c),:,:]),axis=0)
    train_y = np.array([0,1,2,3]).repeat(72-n_test_class)

    test_x=np.concatenate((lei_data[0,(a),:,:],lei_data[1,(a),:,:],
                            lei_data[2,(a),:,:],lei_data[3,(a),:,:]),axis=0)
    test_y = np.array([0,1,2,3]).repeat(n_test_class)
    return train_x,train_y,test_x,test_y

