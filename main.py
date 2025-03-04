import storm_forcast as sf

if __name__ == "__main__":
    # Set training hyperparameters
    hyperparameters = dict(
        root_dir='~/Downloads/data',
        download=False,
        seq_size=20,
        storm_list=['acd'],
        batch_size=1,
        num_kernels=6,
        kernel_size=(3, 3),
        padding=(1, 1),
        frame_size=(366, 366),
        activation="relu",
        num_layers=3,
        lr=1e-4,
        epochs=30,
        surprise=False
    )

    sf.train_model(hyperparameters)
