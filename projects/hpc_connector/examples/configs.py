"""
Example configurations for various HPC clusters.
"""

# SLURM cluster configuration
SLURM_CLUSTER = {
    "name": "slurm-cluster-01",
    "cluster_type": "slurm",
    "ssh": {
        "host": "login.slurm-cluster.edu",
        "port": 22,
        "user": "username",
        "auth_method": "key",
        "key_file": "~/.ssh/id_rsa_slurm",
        "timeout": 30,
        "keepalive_interval": 60,
    },
    "work_dir": "/scratch/username",
    "module_system": "lmod",
    "environment_setup": [
        "source /etc/profile.d/modules.sh",
    ],
    "default_partition": "compute",
    "max_nodes": 64,
    "max_walltime": "72:00:00",
    "data_staging_enabled": True,
    "staging_dir": "/scratch/username/staging",
    "monitoring_interval": 30,
    "features": ["infiniband", "lustre", "gpu"],
}

# PBS/Torque cluster configuration
PBS_CLUSTER = {
    "name": "pbs-cluster-01",
    "cluster_type": "pbs",
    "ssh": {
        "host": "login.pbs-cluster.org",
        "port": 22,
        "user": "username",
        "auth_method": "key",
        "key_file": "~/.ssh/id_rsa_pbs",
    },
    "work_dir": "/home/username/work",
    "default_queue": "batch",
    "max_nodes": 32,
    "max_walltime": "48:00:00",
    "data_staging_enabled": True,
    "features": ["mpi", "openmp"],
}

# LSF cluster configuration
LSF_CLUSTER = {
    "name": "lsf-cluster-01",
    "cluster_type": "lsf",
    "ssh": {
        "host": "login.lsf-cluster.ibm.com",
        "port": 22,
        "user": "username",
        "auth_method": "key",
        "key_file": "~/.ssh/id_rsa_lsf",
    },
    "work_dir": "/gpfs/username",
    "default_queue": "normal",
    "max_nodes": 128,
    "max_walltime": "168:00:00",
    "features": ["gpfs", "spectrum_mpi"],
}

# SGE cluster configuration
SGE_CLUSTER = {
    "name": "sge-cluster-01",
    "cluster_type": "sge",
    "ssh": {
        "host": "login.sge-cluster.university.edu",
        "port": 22,
        "user": "username",
        "auth_method": "key",
        "key_file": "~/.ssh/id_rsa_sge",
    },
    "work_dir": "/data/username",
    "default_queue": "all.q",
    "max_nodes": 48,
    "max_walltime": "24:00:00",
    "features": ["smp", "mpi"],
}

# Job configuration examples
SEQUENTIAL_JOB = {
    "name": "sequential_job",
    "command": "python script.py",
    "work_dir": "/scratch/username/job001",
    "resources": {
        "nodes": 1,
        "cores_per_node": 1,
        "memory_per_node": "4GB",
        "walltime": "1:00:00",
    },
    "modules": ["python/3.9"],
    "environment": {
        "OMP_NUM_THREADS": "1",
    },
    "stdout": "job.out",
    "stderr": "job.err",
}

MPI_JOB = {
    "name": "mpi_simulation",
    "command": "mpirun -np 64 ./simulation.exe",
    "work_dir": "/scratch/username/mpi_job",
    "resources": {
        "nodes": 4,
        "cores_per_node": 16,
        "memory_per_node": "32GB",
        "walltime": "4:00:00",
        "queue": "mpi",
    },
    "modules": ["intelmpi/2019", "mkl/2019"],
    "environment": {
        "I_MPI_FABRICS": "shm:ofi",
    },
}

GPU_JOB = {
    "name": "gpu_training",
    "command": "python train.py --epochs 100 --batch-size 64",
    "work_dir": "/scratch/username/gpu_job",
    "resources": {
        "nodes": 1,
        "cores_per_node": 8,
        "gpus_per_node": 2,
        "memory_per_node": "64GB",
        "walltime": "12:00:00",
        "partition": "gpu",
        "constraints": ["v100"],
    },
    "modules": ["cuda/11.8", "cudnn/8.6", "pytorch/2.0"],
    "environment": {
        "CUDA_VISIBLE_DEVICES": "0,1",
    },
}

ARRAY_JOB = {
    "name": "array_job",
    "command": "python process.py $PBS_ARRAYID",
    "work_dir": "/scratch/username/array_job",
    "resources": {
        "nodes": 1,
        "cores_per_node": 4,
        "walltime": "2:00:00",
    },
    "modules": ["python/3.9"],
}

CHECKPOINT_JOB = {
    "name": "long_simulation",
    "command": "python simulation.py --checkpoint-dir $CHECKPOINT_DIR",
    "work_dir": "/scratch/username/checkpoint_job",
    "resources": {
        "nodes": 8,
        "cores_per_node": 32,
        "memory_per_node": "128GB",
        "walltime": "72:00:00",
    },
    "checkpoint_enabled": True,
    "checkpoint_interval": 3600,
    "checkpoint_dir": "/scratch/username/checkpoint_job/checkpoints",
}

DEPENDENT_JOBS = {
    "preprocessing": {
        "name": "preprocessing",
        "command": "python preprocess.py",
        "work_dir": "/scratch/username/workflow",
        "resources": {
            "nodes": 1,
            "cores_per_node": 8,
            "walltime": "2:00:00",
        },
    },
    "simulation": {
        "name": "simulation",
        "command": "python simulate.py",
        "work_dir": "/scratch/username/workflow",
        "resources": {
            "nodes": 4,
            "cores_per_node": 16,
            "walltime": "24:00:00",
        },
        "dependencies": [],  # Will be set to preprocessing job ID
    },
    "postprocessing": {
        "name": "postprocessing",
        "command": "python postprocess.py",
        "work_dir": "/scratch/username/workflow",
        "resources": {
            "nodes": 1,
            "cores_per_node": 4,
            "walltime": "1:00:00",
        },
        "dependencies": [],  # Will be set to simulation job ID
    },
}
