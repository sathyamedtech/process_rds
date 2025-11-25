import numpy as np
import scipy.signal
import mne
import EntropyHub as EH
from joblib import Parallel, delayed
from logger_config import logging
from auto_config import (CHANNELS, EVENT_DURATION, FILTER_MIN_DISTANCE_FROM_START, FILTER_MIN_DISTANCE_FROM_END,
                         DEFINED_BANDS, N_FFT, N_OVERLAP)

class Feature_Extraction:
    """
    Class to extract features from EEG signal and save to a database.

    Methods:
    downsample(raw)
        Downsamples EEG data to desired rate. Also creates montage and renames channels

    create_events(raw_downsampled)
        creates events to use for epoching

    create_epochs(raw_downsampled, events, event_id)
        created epochs from downsampled data and events

    power_ratio(epochs)
        calculates the power ratio

    coherence(raw_data)
        calculates coherence

    complexity(data, fuzzy_embedding=3)
        calculates complexity
    """

    def __init__(self):
        # Initialise class
        pass

    def downsample(self, raw, freq, montage):
        # Rename channels - to clean the channel names so they are usable!
        new_names = dict(
            (ch_name, ch_name.rstrip(".").replace("EEG ", "").replace("-Ref", "").replace("Z", "z").replace("FP", "Fp"))
            for ch_name in raw.ch_names
        )
        raw.rename_channels(new_names)

        # Pick only the specified channels (use all 19 which may be relevant)
        # Note: This is applied even though it looks unused in the code!!
        raw_selected = raw.pick_channels(CHANNELS)
        
        # Downsample raw with mne - input integer is desired sampling rate
        downsampled_rate = freq # Normal setting is 250Hz based on email from Kartheka@BB on 05/02/25
        raw_downsampled = raw.copy().resample(downsampled_rate)
        # Set montage - using standard alphabetic suggested by BB
        montage = mne.channels.make_standard_montage(montage)
        raw_downsampled.set_montage(montage)
        raw_downsampled.set_eeg_reference(projection=True)  # needed for inverse modeling
        return(raw_downsampled)

    def create_events(self, raw_downsampled):
        # Create events
        # At eg. 500Hz, 1 second = 50 events, so want the events duration short to capture full view of data
        interval_seconds = EVENT_DURATION
        # Get the total duration of the raw data in seconds
        total_duration = raw_downsampled.times[-1]
        # Create events every "interval_seconds" seconds
        event_times = np.arange(0, total_duration, interval_seconds)
        # Convert event times to samples
        event_samples = (event_times * raw_downsampled.info["sfreq"]).astype(int)
        # Assign unique event_id values for different types of events
        event_id_values = np.arange(1, len(event_times) + 1)
        # Filter cut off start/end events to avoid bad epochs - Set minimum distance from start (here it is 2 seconds)
        min_distance_from_start = int(FILTER_MIN_DISTANCE_FROM_START * raw_downsampled.info["sfreq"])
        # Set min distance from end - here it is 3 seconds, to add to the interval length
        min_distance_from_end = int(FILTER_MIN_DISTANCE_FROM_END * raw_downsampled.info["sfreq"])
        # Filter out events that are too close to the start or end
        valid_event_samples = []
        valid_event_time = []
        valid_event_ids = []
        for i, event_sample in enumerate(event_samples):
            if event_sample > min_distance_from_start and event_sample < (len(raw_downsampled) - min_distance_from_end):
                valid_event_samples.append(event_sample)
                valid_event_time.append(event_times[i])
                valid_event_ids.append(event_id_values[i])
        
        # Ensure events is a numpy array of integers using sanples, zeros for event type, and ids
        events = np.column_stack((valid_event_samples, np.zeros_like(valid_event_samples), valid_event_ids)).astype(int)
        
        # Create annotations for events with onset, zeros for duration, and description
        annotations = mne.Annotations(onset=valid_event_time, duration=np.zeros_like(valid_event_time), description="Event")
        
        # Apply annotations to raw data
        raw_downsampled.set_annotations(annotations)
        
        # Create dictionary mapping event labels to event id values
        event_id = {f"Event_{i + 1}": code for i, code in enumerate(valid_event_ids)}
        
        return events, event_id, raw_downsampled
    
    def create_epochs(self, raw_downsampled, events, event_id):
        # Create epochs using events and event duration - tmin and tmax should not be set so events overlap, so pay attention to length of events
        # because of this epochs start 2 sec before an event and end 3 sec after
        epochs = mne.Epochs(raw_downsampled, events, event_id=event_id, tmin=-FILTER_MIN_DISTANCE_FROM_START, tmax=FILTER_MIN_DISTANCE_FROM_END, baseline=(None,0), preload=True)
        # This flags if drop log is needed (if epoch is empty)
        if len(epochs) == 0:
            raise RuntimeError("No epochs found. Please check the drop log for details.")
        # Convert epochs to a numpy array (n_epochs, n_channels, n_times). This is the data structure that is later controlling the size of eg. fuzzy
        data = epochs.get_data(copy=True)
        raw_data = raw_downsampled.get_data()
        return(raw_data, data, epochs)

    def power_ratio(self, epochs, bands=None):
        if bands is None:
            bands = DEFINED_BANDS

        # Initialise arrays to store power for each band and to store psd values
        power = {band: np.zeros(len(epochs.ch_names)) for band in bands}
        psd_values = {}

        # Calculate psd for each channel and each frequency band using welch's method
        for band, (fmin, fmax) in bands.items():
            spectrum = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=N_FFT, n_overlap=N_OVERLAP)
            # get psd and freqs - currently freqs is unutilised but might be useful later
            psd, freqs = spectrum.get_data(return_freqs=True)
            power[band] = psd.mean(axis=2).mean(axis=0) # Average across epochs and frequency bins
            # Sum for each band
            psd_values[band] = np.sum(power[band])

        # Calculate the ratio (alpha + beta) / (theta + delta)
        alpha = np.sum(power["alpha"])
        beta = np.sum(power["beta"])
        delta = np.sum(power["delta"])
        theta = np.sum(power["theta"])
        combined_ratio = (alpha+beta)/(delta+theta)

        # Generate power ratios for each channel
        channel_ratios = {}
        for ch_idx, ch_name in enumerate(epochs.ch_names):
            alpha_ch = power["alpha"][ch_idx]
            beta_ch = power["beta"][ch_idx]
            delta_ch = power["delta"][ch_idx]
            theta_ch = power["theta"][ch_idx]
            channel_ratios[ch_name] = (alpha_ch + beta_ch) / (delta_ch + theta_ch)

        return(channel_ratios, combined_ratio, psd_values)

    def full_coherence(self, raw_data):
        # Initialise summary statistics lists
        freqs = []
        coh = []
        pairs = []
        coh_median = []
        coh_min = []
        coh_max = []
        coh_std_dev = []
        coh_mean = []

        # Set of channel labels
        chan_labels = CHANNELS
        
        # Generate coherence values for every combination of channels
        for i in range(len(chan_labels)):
            for j in range(len(chan_labels)):
                if i < raw_data.shape[0] and j < raw_data.shape[0]:
                    sig1 = raw_data[i, :]
                    sig2 = raw_data[j, :]
                    f, cxy = scipy.signal.coherence(sig1, sig2)
                    pair_name = f"{chan_labels[i]}-{chan_labels[j]}"
                    
                    freqs.append(f)
                    coh.append(cxy)
                    pairs.append(pair_name)
                    coh_min.append(np.min(cxy))
                    coh_max.append(np.max(cxy))
                    coh_median.append(np.median(cxy))
                    coh_std_dev.append(np.std(cxy))
                    coh_mean.append((pair_name, np.mean(cxy)))

        return freqs, coh, pairs, coh_min, coh_max, coh_median, coh_std_dev, coh_mean

    def complexity(self, data, fuzzy_embedding=3): # Smaller embedding runs faster (2 minimum)
        chan_labels = CHANNELS
        if data.shape[1] != len(chan_labels):
            logging.error(f"Mismatch: data has {data.shape[1]} channels, but chan_labels has {len(chan_labels)} labels.")
        # Use a placeholder to determine the shape of fuzzy (using random data)
        sample_fuzzy, _ = EH.MSEn(np.random.randn(2501), EH.MSobject(EnType="FuzzEn", m=fuzzy_embedding))
        channel_complexities = np.zeros((data.shape[1], sample_fuzzy.shape[0]))

        # function used to enable parallel/delayed job processing to speed up complexity calculation (batch processing)
        def process_epoch(epoch, i):
            # Normalise epoch
            normalised_epoch = (epoch[i] - np.mean(epoch[i])) / np.std(epoch[i])
            # Create multiscale fuzzy object
            ms_object = EH.MSobject(EnType="FuzzEn", m=fuzzy_embedding)
            # Use fuzzy object to find multiscale fuzzy entropy
            fuzzy, _ = EH.MSEn(normalised_epoch, ms_object) #Scales=fuzzy_embedding
            return fuzzy

        # This is the parallel jobs code
        channel_complexity_data = {}
        for i, channel_name in enumerate(chan_labels):
            fuzzy_values = Parallel(n_jobs=-1)(delayed(process_epoch)(epoch, i) for epoch in data)
            channel_complexities[i] = np.mean(fuzzy_values, 0)
            channel_complexity_data[channel_name] = channel_complexities[i]

        return channel_complexity_data
