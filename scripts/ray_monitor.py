#!/usr/bin/env python3
"""
Ray Cluster Monitor
Simple monitoring script for Ray cluster when Dashboard is not available
"""

import ray
import time
import psutil
from datetime import datetime

def get_cluster_info():
    """Get Ray cluster information"""
    try:
        if not ray.is_initialized():
            ray.init(address="auto")
        
        # Get cluster resources
        resources = ray.cluster_resources()
        
        # Get nodes
        nodes = ray.nodes()
        
        return {
            "resources": resources,
            "nodes": nodes,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e)}

def get_system_info():
    """Get system information"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def print_cluster_status():
    """Print cluster status"""
    info = get_cluster_info()
    system = get_system_info()
    
    print("=" * 60)
    print(f"Ray Cluster Monitor - {info.get('timestamp', 'Unknown')}")
    print("=" * 60)
    
    if "error" in info:
        print(f"❌ Error: {info['error']}")
        return
    
    # Cluster resources
    print("\n📊 Cluster Resources:")
    resources = info.get("resources", {})
    for resource, value in resources.items():
        if isinstance(value, (int, float)):
            if "memory" in resource.lower():
                print(f"  {resource}: {value / (1024**2):.1f} MB")
            elif "cpu" in resource.lower():
                print(f"  {resource}: {value}")
            else:
                print(f"  {resource}: {value}")
    
    # Node information
    print(f"\n🖥️  Nodes: {len(info.get('nodes', []))}")
    for i, node in enumerate(info.get("nodes", []), 1):
        print(f"  Node {i}:")
        print(f"    Address: {node.get('NodeManagerAddress', 'Unknown')}")
        print(f"    Alive: {node.get('Alive', False)}")
        node_resources = node.get('Resources', {})
        print(f"    Resources: {len(node_resources)} types")
    
    # System information
    print(f"\n💻 System Info:")
    print(f"  CPU Usage: {system['cpu_percent']:.1f}%")
    print(f"  Memory Usage: {system['memory_percent']:.1f}%")
    print(f"  Disk Usage: {system['disk_usage']:.1f}%")

def monitor_ray_cluster(interval=5):
    """Monitor Ray cluster continuously"""
    try:
        while True:
            print_cluster_status()
            print(f"\n⏰ Next update in {interval} seconds...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ray Cluster Monitor")
    parser.add_argument("--interval", type=int, default=5, 
                       help="Monitoring interval in seconds")
    parser.add_argument("--once", action="store_true",
                       help="Run once and exit")
    
    args = parser.parse_args()
    
    if args.once:
        print_cluster_status()
    else:
        monitor_ray_cluster(args.interval)
