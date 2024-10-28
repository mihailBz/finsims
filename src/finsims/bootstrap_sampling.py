import numpy as np


def stationary_block_bootstrap(
    historical_data, number_required=200, exp_block_size=20, num_paths=1000
):
    N = len(historical_data)
    all_bootstrap_samples = []

    for _ in range(num_paths):
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
                    break

        # Append the generated path to the list of all paths
        all_bootstrap_samples.append(bootstrap_samples[:number_required])

    return all_bootstrap_samples
