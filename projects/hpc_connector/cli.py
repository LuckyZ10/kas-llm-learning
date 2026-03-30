"""
HPC Connector - CLI Interface

Command-line interface for HPC Connector operations.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from hpc_connector import HPCClient, ClusterConfig
from hpc_connector.core.cluster import ClusterManager


def cmd_clusters(args):
    """List configured clusters."""
    manager = ClusterManager()
    
    config_file = args.config or "clusters.yaml"
    if Path(config_file).exists():
        manager.load_from_file(config_file)
    
    print("Configured Clusters:")
    print("-" * 50)
    for name in manager.list_clusters():
        config = manager.get(name)
        print(f"  {name}")
        print(f"    Type: {config.cluster_type.value}")
        print(f"    Host: {config.ssh.host}")
        print(f"    User: {config.ssh.user}")
        print()


async def cmd_submit(args):
    """Submit a job."""
    config = ClusterConfig.from_dict({
        "name": args.cluster,
        "cluster_type": args.scheduler,
        "ssh": {
            "host": args.host,
            "port": args.port,
            "user": args.user,
            "auth_method": "key",
            "key_file": args.key,
        },
    })
    
    async with HPCClient(config) as client:
        job_spec = {
            "name": args.name,
            "command": args.command,
            "work_dir": args.work_dir,
            "resources": {
                "nodes": args.nodes,
                "cores_per_node": args.cores,
                "walltime": args.walltime,
            },
        }
        
        if args.queue:
            job_spec["resources"]["queue"] = args.queue
        
        job_id = await client.submit_job(job_spec)
        print(f"Job submitted: {job_id}")
        
        if args.wait:
            print("Waiting for job completion...")
            result = await client.wait_for_job(job_id)
            print(f"Status: {result.status.value}")
            print(f"Exit code: {result.exit_code}")


async def cmd_status(args):
    """Check job status."""
    config = ClusterConfig.from_dict({
        "name": args.cluster,
        "cluster_type": args.scheduler,
        "ssh": {
            "host": args.host,
            "port": args.port,
            "user": args.user,
            "auth_method": "key",
            "key_file": args.key,
        },
    })
    
    async with HPCClient(config) as client:
        status = await client.get_job_status(args.job_id)
        print(f"Job {args.job_id}: {status.value}")


async def cmd_list(args):
    """List jobs."""
    config = ClusterConfig.from_dict({
        "name": args.cluster,
        "cluster_type": args.scheduler,
        "ssh": {
            "host": args.host,
            "port": args.port,
            "user": args.user,
            "auth_method": "key",
            "key_file": args.key,
        },
    })
    
    async with HPCClient(config) as client:
        jobs = await client.list_jobs(user=args.user)
        
        print(f"{'Job ID':<12} {'Name':<20} {'Status':<12} {'Queue':<15}")
        print("-" * 65)
        for job in jobs[:args.limit]:
            print(f"{job['job_id']:<12} {job.get('name', 'N/A'):<20} "
                  f"{job['status']:<12} {job.get('queue', 'N/A'):<15}")


async def cmd_queues(args):
    """List queues."""
    config = ClusterConfig.from_dict({
        "name": args.cluster,
        "cluster_type": args.scheduler,
        "ssh": {
            "host": args.host,
            "port": args.port,
            "user": args.user,
            "auth_method": "key",
            "key_file": args.key,
        },
    })
    
    async with HPCClient(config) as client:
        queues = await client.get_queues()
        
        print(f"{'Queue':<20} {'State':<10} {'Total':<8} {'Running':<10} {'Queued':<10}")
        print("-" * 65)
        for queue in queues:
            print(f"{queue['name']:<20} {queue['state']:<10} "
                  f"{queue.get('total_nodes', 'N/A'):<8} "
                  f"{queue.get('jobs_running', 'N/A'):<10} "
                  f"{queue.get('jobs_queued', 'N/A'):<10}")


def cmd_upload(args):
    """Upload files."""
    print("Upload functionality - implement with actual connection")


def cmd_download(args):
    """Download files."""
    print("Download functionality - implement with actual connection")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HPC Connector CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hpc-connector clusters --config clusters.yaml
  hpc-connector submit --host login.cluster.org --user myuser \\
                       --command "python train.py" --nodes 2 --cores 8
  hpc-connector status --job-id 12345
  hpc-connector list --limit 20
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Configuration file path'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # clusters command
    clusters_parser = subparsers.add_parser('clusters', help='List clusters')
    clusters_parser.set_defaults(func=cmd_clusters)
    
    # submit command
    submit_parser = subparsers.add_parser('submit', help='Submit a job')
    submit_parser.add_argument('--host', required=True, help='Cluster login host')
    submit_parser.add_argument('--port', type=int, default=22, help='SSH port')
    submit_parser.add_argument('--user', '-u', required=True, help='Username')
    submit_parser.add_argument('--key', '-k', default='~/.ssh/id_rsa', help='SSH key file')
    submit_parser.add_argument('--scheduler', '-s', default='slurm', 
                               choices=['slurm', 'pbs', 'torque', 'lsf', 'sge'],
                               help='Scheduler type')
    submit_parser.add_argument('--cluster', default='default', help='Cluster name')
    submit_parser.add_argument('--name', '-n', required=True, help='Job name')
    submit_parser.add_argument('--command', '-C', required=True, help='Command to run')
    submit_parser.add_argument('--work-dir', '-w', default='.', help='Working directory')
    submit_parser.add_argument('--nodes', '-N', type=int, default=1, help='Number of nodes')
    submit_parser.add_argument('--cores', '-c', type=int, default=1, help='Cores per node')
    submit_parser.add_argument('--walltime', '-t', default='1:00:00', help='Walltime')
    submit_parser.add_argument('--queue', '-q', help='Queue/Partition')
    submit_parser.add_argument('--wait', action='store_true', help='Wait for completion')
    submit_parser.set_defaults(func=lambda args: asyncio.run(cmd_submit(args)))
    
    # status command
    status_parser = subparsers.add_parser('status', help='Check job status')
    status_parser.add_argument('--job-id', '-j', required=True, help='Job ID')
    status_parser.add_argument('--host', required=True, help='Cluster login host')
    status_parser.add_argument('--port', type=int, default=22, help='SSH port')
    status_parser.add_argument('--user', '-u', required=True, help='Username')
    status_parser.add_argument('--key', '-k', default='~/.ssh/id_rsa', help='SSH key file')
    status_parser.add_argument('--scheduler', '-s', default='slurm',
                               choices=['slurm', 'pbs', 'torque', 'lsf', 'sge'])
    status_parser.add_argument('--cluster', default='default', help='Cluster name')
    status_parser.set_defaults(func=lambda args: asyncio.run(cmd_status(args)))
    
    # list command
    list_parser = subparsers.add_parser('list', help='List jobs')
    list_parser.add_argument('--host', required=True, help='Cluster login host')
    list_parser.add_argument('--port', type=int, default=22, help='SSH port')
    list_parser.add_argument('--user', '-u', required=True, help='Username')
    list_parser.add_argument('--key', '-k', default='~/.ssh/id_rsa', help='SSH key file')
    list_parser.add_argument('--scheduler', '-s', default='slurm',
                             choices=['slurm', 'pbs', 'torque', 'lsf', 'sge'])
    list_parser.add_argument('--cluster', default='default', help='Cluster name')
    list_parser.add_argument('--limit', '-l', type=int, default=20, help='Maximum jobs to show')
    list_parser.set_defaults(func=lambda args: asyncio.run(cmd_list(args)))
    
    # queues command
    queues_parser = subparsers.add_parser('queues', help='List queues')
    queues_parser.add_argument('--host', required=True, help='Cluster login host')
    queues_parser.add_argument('--port', type=int, default=22, help='SSH port')
    queues_parser.add_argument('--user', '-u', required=True, help='Username')
    queues_parser.add_argument('--key', '-k', default='~/.ssh/id_rsa', help='SSH key file')
    queues_parser.add_argument('--scheduler', '-s', default='slurm',
                               choices=['slurm', 'pbs', 'torque', 'lsf', 'sge'])
    queues_parser.add_argument('--cluster', default='default', help='Cluster name')
    queues_parser.set_defaults(func=lambda args: asyncio.run(cmd_queues(args)))
    
    # upload command
    upload_parser = subparsers.add_parser('upload', help='Upload files')
    upload_parser.add_argument('local', help='Local file path')
    upload_parser.add_argument('remote', help='Remote file path')
    upload_parser.set_defaults(func=cmd_upload)
    
    # download command
    download_parser = subparsers.add_parser('download', help='Download files')
    download_parser.add_argument('remote', help='Remote file path')
    download_parser.add_argument('local', help='Local file path')
    download_parser.set_defaults(func=cmd_download)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
