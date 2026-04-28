#!/usr/bin/env python3
"""
Ray Cluster Startup Script
Start and manage Ray cluster for distributed RAG processing
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ray
from ray._private.ray_constants import DEFAULT_DASHBOARD_PORT as RAY_DEFAULT_DASHBOARD_PORT

def start_head_node():
    """Start Ray head node"""
    print("Starting Ray head node...")
    
    try:
        # Start Ray head node
        ray.init(
            address=None,  # Start new cluster
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            log_to_driver=True,
            include_dashboard=True,
            num_cpus=4,
            num_gpus=0,
            resources={"head_node": 1}
        )
        
        # Get cluster info
        cluster_info = ray.cluster_resources()
        dashboard_url = f"http://localhost:{RAY_DEFAULT_DASHBOARD_PORT}"
        
        print(f"Ray head node started successfully!")
        print(f"Dashboard URL: {dashboard_url}")
        print(f"Cluster resources: {cluster_info}")
        
        return True
        
    except Exception as e:
        print(f"Failed to start Ray head node: {e}")
        return False

def start_worker_nodes(num_workers: int = 2):
    """Start Ray worker nodes"""
    print(f"Starting {num_workers} Ray worker nodes...")
    
    try:
        # Get head node address
        head_address = ray.get_dashboard_url().split("//")[1].split(":")[0]
        
        for i in range(num_workers):
            print(f"Starting worker node {i+1}...")
            
            # Start worker in subprocess
            worker_cmd = [
                sys.executable, "-m", "ray", "start",
                "--address", f"{head_address}:6379",
                "--num-cpus", "8",
                "--num-gpus", "1",
                "--resources", f"worker_node_{i+1}=1"
            ]
            
            process = subprocess.Popen(worker_cmd, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
            
            # Wait a bit for worker to start
            time.sleep(2)
            
            if process.poll() is None:
                print(f"Worker node {i+1} started successfully")
            else:
                print(f"Worker node {i+1} failed to start")
                stdout, stderr = process.communicate()
                print(f"Error: {stderr.decode()}")
        
        return True
        
    except Exception as e:
        print(f"Failed to start worker nodes: {e}")
        return False

def check_cluster_status():
    """Check Ray cluster status"""
    try:
        if not ray.is_initialized():
            print("Ray cluster is not initialized")
            return False
        
        # Get cluster info
        nodes = ray.nodes()
        resources = ray.cluster_resources()
        
        print(f"Cluster Status:")
        print(f"  Head node: {nodes[0]['NodeManagerAddress']}")
        print(f"  Total nodes: {len(nodes)}")
        print(f"  Total resources: {resources}")
        
        # Check each node
        for i, node in enumerate(nodes):
            print(f"  Node {i+1}:")
            print(f"    Address: {node['NodeManagerAddress']}")
            print(f"    Alive: {node['Alive']}")
            print(f"    Resources: {node['Resources']}")
        
        return True
        
    except Exception as e:
        print(f"Failed to check cluster status: {e}")
        return False

def shutdown_cluster():
    """Shutdown Ray cluster"""
    print("Shutting down Ray cluster...")
    
    try:
        if ray.is_initialized():
            ray.shutdown()
            print("Ray cluster shutdown successfully")
        else:
            print("Ray cluster is not running")
        
        return True
        
    except Exception as e:
        print(f"Failed to shutdown cluster: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Ray Cluster Manager")
    parser.add_argument("action", choices=["start", "stop", "status", "start-workers"],
                       help="Action to perform")
    parser.add_argument("--workers", type=int, default=2,
                       help="Number of worker nodes to start")
    
    args = parser.parse_args()
    
    if args.action == "start":
        success = start_head_node()
        if success:
            print("Waiting for cluster to stabilize...")
            time.sleep(5)
            check_cluster_status()
    
    elif args.action == "start-workers":
        success = start_worker_nodes(args.workers)
        if success:
            time.sleep(3)
            check_cluster_status()
    
    elif args.action == "status":
        check_cluster_status()
    
    elif args.action == "stop":
        shutdown_cluster()
    
    else:
        print(f"Unknown action: {args.action}")

if __name__ == "__main__":
    main()
