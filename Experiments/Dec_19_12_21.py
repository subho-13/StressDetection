#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy
import pandas
import bisect
import tensorflow


# In[16]:


def get_segment_info(labels):
    segment_info = []

    # Segment info
    curr_label = -1
    begin = -1

    for i in range(len(labels)):
        if curr_label != labels[i]:
            segment_info.append((curr_label, begin, i - 1))

            begin = i
            curr_label = labels[i]

    return segment_info


# In[17]:


EPSILON = 1e-6


# In[18]:


def is_within_range(mean_val, curr_val, max_relative_deviation):
    if abs(mean_val - curr_val) / max(EPSILON, abs(mean_val)) < max_relative_deviation:
        return True

    return False


# In[19]:


def get_amp_freq(signal_segment, sampling_freq, freq_range):
    # Apply Hamming window to the signal segment
    window = numpy.hamming(len(signal_segment))
    corrected_signal_segment = signal_segment * window

    # Calculating amps and freqs
    fft_coefficients = numpy.fft.fft(signal_segment)
    amps = 2 / len(signal_segment) * numpy.abs(fft_coefficients)
    freqs = numpy.fft.fftfreq(len(amps), 1 / sampling_freq)

    # Taking the left half of amps and freqs
    amps = amps[0:len(amps) // 2]
    freqs = freqs[0:len(freqs) // 2]

    # Sorting amps and freqs by freqs
    sort_freqs = numpy.argsort(freqs)
    amps = amps[sort_freqs]
    freqs = freqs[sort_freqs]

    # Filter section within given freq_range
    left_index = bisect.bisect_left(freqs, freq_range[0])
    right_index = bisect.bisect_right(freqs, freq_range[1])
    amps = amps[left_index: right_index + 1]
    freqs = freqs[left_index: right_index + 1]

    # Sorting amps and freqs by amps
    sort_amps = numpy.argsort(amps)
    amps = amps[sort_amps]
    freqs = freqs[sort_amps]
    
    print(amps, freqs)

    return amps, freqs


# In[20]:


def get_segment_feature(signal_segment, sampling_freq, freq_range, max_relative_deviation, num_features):
    amps, freqs = get_amp_freq(signal_segment, sampling_freq, freq_range)

    amp_sum = amps[0]
    freq_sum = freqs[0]
    n = 1
    features = []

    for i in range(len(amps)):
        if num_features == 0:
            break

        amp = amps[i]
        freq = freqs[i]

        is_ok = is_within_range(freq_sum / n, freq, max_relative_deviation)
        is_ok &= is_within_range(amp_sum / n, amp, max_relative_deviation)

        if is_ok:
            amp_sum += amp
            freq_sum += freq
            n += 1
        else:
            features.append((amp_sum / n, freq_sum / n))
            num_features -= 1

            amp_sum = amp
            freq_sum = freq
            n = 1

    return features


# In[21]:


TRANSIENT = 0
BASELINE = 1
STRESS = 2
AMUSEMENT = 3
MEDITATION = 4
IGNORE = 5


# In[22]:


def get_signal_features(signal, segment_info, sampling_freq, window_len, overlap, freq_range, max_relative_deviation, num_features) :
    sub_segment_len = window_len * sampling_freq

    signal_features = []
    labels = []
    
    baseline_average = 0
    
    for curr_segment_info in segment_info :
        label = curr_segment_info[0]
        begin = curr_segment_info[1]
        end = curr_segment_info[2]
        
        if label >= IGNORE or label == TRANSIENT:
            continue
        
        segment = signal[begin:end]      
            
        if label == BASELINE :
            baseline_average = numpy.mean(segment)
        
        segment -= baseline_average
        
        curr_segment_len = end - begin        
        while curr_segment_len >= sub_segment_len:
            sub_segment = segment[int(curr_segment_len - sub_segment_len) : curr_segment_len]
            curr_segment_len -= int((1 - overlap) * sub_segment_len)
            
            segment_features = get_segment_feature(sub_segment, sampling_freq, freq_range, max_relative_deviation, num_features)            
            
            signal_features.append(segment_features)
            labels.append(label)
            
            print(curr_segment_len)
        
    return signal_features, labels


# In[23]:


CHEST_SAMPLING_FREQ = 700
OVERLAP = 0
MAX_RELATIVE_DEVIATION = 0.01
NUM_FEATURES = 10

FREQ_RANGE = {
    "ECG" : (0.1, 150),
    "EMG" : (0, 1000),
    "EDA" : (0, 4),
    "Resp" : (0, 2),
}

WINDOW_LEN = {
    "ECG" : 30,
    "EMG" : 30,
    "EDA" : 30,
    "Resp" : 60,
}

ALIGN_LEN = 30


# In[24]:


def align_list(feature_set, window_len, align_len) :
    tmp = 0
    i = 0
    
    new_set = []
    while i < len(feature_set) :
        new_set.append(feature_set[i])
        tmp += align_len
        i = int(tmp/window_len)
        
    return new_set


# In[25]:


PATH = '../WESAD/'
SUBJECTS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '13', '14', '15', '16', '17']
CHEST_SIGNALS = ['Resp', 'ECG', 'EMG', 'EDA', 'Temp', 'ACC']


# In[26]:


def read_subject_data(subject) :
    path = PATH + 'S' + subject + '/S' + subject + '.pkl'
    subject = pandas.read_pickle(path)
    
    return subject


# In[27]:


def get_flattened_feature_set(feature_set, size) :
    remove_index_set = []
    
    for i in range(size) :
        tmp_label = None
        
        for signal_type in feature_set:
            if tmp_label is None :
                tmp_label = feature_set[signal_type]['labels'][i]
            elif feature_set[signal_type]['labels'][i] != tmp_label :
                remove_index_set.append(i)
                
    for index in remove_index_set :
        for signal_type in feature_set :
            feature_set[signal_type]['labels'].pop(index)
            feature_set[signal_type]['signal_features'].pop(index)
            
    common_labels = None
    flattened_feature_set = {}
    
    for signal_type in feature_set :
        if common_labels is None :
            common_labels = feature_set[signal_type]['labels']
            
        flattened_feature_set[signal_type] = feature_set[signal_type]['signal_features']
    
    flattened_feature_set['labels'] = common_labels
    
    return flattened_feature_set


# In[28]:


def one_hot_encode(labels) :
    new_labels = []
    
    for label in labels :
        new_label = []
        
        for i in range(4) :
            if i == label - 1 :
                new_label.append(1)
            else :
                new_label.append(0)
                
        new_labels.append(new_label)
        
    return new_labels


# In[29]:


def get_subject_feature(subject) :
    subject_data = read_subject_data(subject)
    segment_info = get_segment_info(subject_data['label'])
    
    feature_set = {}
    min_len = len(subject_data['label'])
    for signal_type in CHEST_SIGNALS :
        
        if signal_type == 'Temp' or signal_type == 'ACC' :
            continue
            
        signal = subject_data['signal']['chest'][signal_type]

        freq_range = FREQ_RANGE[signal_type]
        window_len = WINDOW_LEN[signal_type]
        
        signal_features, labels = get_signal_features(signal, segment_info, CHEST_SAMPLING_FREQ, window_len, OVERLAP, freq_range, MAX_RELATIVE_DEVIATION, NUM_FEATURES)
        
        aligned_signal_features = align_list(signal_features, window_len, ALIGN_LEN)
        aligned_labels = align_list(labels, window_len, ALIGN_LEN)
        
        feature_set[signal_type] =  {
            'signal_features' : aligned_signal_features,
            'labels' : aligned_labels
        }
        
        min_len = min(min_len, len(aligned_signal_features))
        print("Feature :: ", signal_type)
        
    for signal_type in feature_set :
        feature_set[signal_type]['signal_features'] = feature_set[signal_type]['signal_features'][:min_len]
        feature_set[signal_type]['labels'] = feature_set[signal_type]['labels'][:min_len]
        
    flattened_feature_set = get_flattened_feature_set(feature_set, min_len)
    print("Subject :: ", subject)
    
    return flattened_feature_set


# In[30]:


def generate_dataset() :
    dataset = {}
    
    for subject in SUBJECTS :
        subject_signal_feature = get_subject_feature(subject)
        subject_signal_feature['labels'] = one_hot_encode(subject_signal_feature['labels'])
        dataset[subject] = subject_signal_feature
            
    return dataset


# In[31]:


def generate_lou_train_test_dataset(dataset, subject) :
    test_dataset = dataset[subject]
    
    train_dataset = {}
    
    for train_subject in SUBJECTS :
        if train_subject == subject :
            continue
            
        for category in dataset[train_subject] :
            if category not in train_dataset :
                train_dataset[category] = []
            
            train_dataset[category] += dataset[train_subject][category]
            
    return train_dataset, test_dataset


# In[ ]:


dataset = generate_dataset()


# In[ ]:





# In[ ]:




