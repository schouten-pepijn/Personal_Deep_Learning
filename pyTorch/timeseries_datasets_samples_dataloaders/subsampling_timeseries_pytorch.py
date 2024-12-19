import torch
from torch.utils.data import Dataset, Sampler

# for time series subsampling
class TimeSeriesSubsampleDataset(Dataset):
    def __init__(self, timeseries, input_len, target_len):
        """
        A dataset for extracting sub-sequences from a larger time series.

        Args:
            timeseries (iterable): The complete time series data.
            input_len (int): The length of the input sequence.
            target_len (int): The length of the target sequence.
        """
        self.timeseries = timeseries
        self.input_len = input_len
        self.target_len = target_len
        self.total_len = input_len + target_len
     
    def __len__(self):
        return len(self.timeseries) - self.total_len + 1

    def __getitem__(self, idx):
        """
        Extracts a sub-sequence from the time series.
        """
        input_seq = self.timeseries[idx:idx + self.input_len]
        target_seq = self.timeseries[idx + self.input_len:idx + self.input_len + self.target_len]
        return input_seq, target_seq
  

# for time series forecasting
class TimeseriesSubsampler(Sampler):
    def __init__(self, data_source, num_samples=None, replacement=None):
        """
        A custom sampler for random subsampling from a dataset.
        
        Args:
            data_source (iterable): The data to be subsampled.
            num_samples (int): The number of samples to draw from the data source.
                If None, will sample the entire data source.
            replacement (bool): Whether sampling is with replacement.
        """
        self.data_source = data_source
        self.data_len = len(data_source)
        self.num_samples = num_samples or len(data_source)
        self.replacement = replacement
        
    def __iter__(self):
        idx = torch.randint(
            low=0,
            high=self.data_len,
            size=(self.num_samples,),
            generator=None,
            dtype=torch.int64
        ) if self.replacement else torch.randperm(self.data_len)[:self.num_samples]
        return iter(idx)
    
    def __len__(self):
        return self.num_samples
        

# for time series classification
class StratifiedTimeseriesSubsampler(Sampler):
    def __init__(self, data_source, num_samples=None, replacement=None, stratify=None):
        """
        Stratified subsampler for time series data.
        
        Args:
            data_source (iterable): The data to be subsampled.
            num_samples (int): The number of samples to draw from the data source.
                If None, will sample the entire data source.
            replacement (bool): Whether sampling is with replacement.
            stratify (list): A list of labels for the data source, to be used for
                stratified sampling.
        """
        self.data_source = data_source
        self.data_len = len(data_source)
        self.num_samples = num_samples or len(data_source)
        self.replacement = replacement
        
        
        if stratify is None or len(stratify) != self.data_len:
            raise ValueError("Stratify must be a list of the same length as the dataset")
        self.stratify = stratify
        
        # calculate the number of samples per strata
        self.strata_indices = self._compute_strata_indices()
        self.strata_sample_counts = self._compute_sample_counts()
        
    def _compute_strata_indices(self):
        """
        Computes the indices for each stratum based on the stratify labels.
        """
        strata_indices = {}
        for idx, label in enumerate(self.stratify.tolist()):
            if label not in strata_indices:
                strata_indices[label] = []
            strata_indices[label].append(idx)
        return strata_indices
    
    def _compute_sample_counts(self):
        """
        Computes the number of samples for each stratum, based on the proportion of
        each label in the data source.
        """
        label_counts = torch.bincount(self.stratify)
        total_labels = label_counts.sum()
        proportions = label_counts.float() / total_labels
        return (proportions * self.num_samples).long()
    
    def __iter__(self):
        sampled_indices = []
        for label, indices in self.strata_indices.items():
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            if self.replacement:
                sampled = torch.multinomial(
                    torch.ones(len(indices_tensor)),
                    num_samples=self.strata_sample_counts[label].item(),
                    replacement=True
                )
            else:
                sampled = torch.randperm(
                    len(indices_tensor))[
                        :self.strata_sample_counts[label].item()]
            
            sampled_indices.extend(indices_tensor[sampled].tolist())
        
        shuffled_indices = torch.randperm(len(sampled_indices))
        
        return iter([sampled_indices[i] for i in shuffled_indices])