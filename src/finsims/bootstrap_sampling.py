import numpy as np


def stationary_block_bootstrap(historical_data, number_required=100, exp_block_size=20):
    N = len(historical_data)
    bootstrap_samples = []

    while len(bootstrap_samples) < number_required:
        # Choose a random starting index between 0 and N-1
        index = np.random.randint(0, N)

        # Sample block size from a geometric distribution
        blocksize = np.random.geometric(1 / exp_block_size)

        # Generate the block of samples
        for i in range(blocksize):
            # Wrap around if the index exceeds the data length
            sample_index = (index + i) % N
            bootstrap_samples.append(historical_data[sample_index])

            # Stop if we have enough samples
            if len(bootstrap_samples) == number_required:
                return bootstrap_samples

    return bootstrap_samples
