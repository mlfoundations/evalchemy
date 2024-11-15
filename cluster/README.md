# Ray Cluster Configuration

This document explains the purpose and functionality of two key YAML files used to set up and manage a Ray cluster on Kubernetes.

## autoscaler.yaml

The `autoscaler.yaml` file defines the configuration for a Ray cluster with built-in autoscaling capabilities. Here's a breakdown of its main components:

1. **RayCluster Resource**: This custom resource defines the Ray cluster configuration.

2. **Autoscaling Configuration**:
   - `enableInTreeAutoscaling: true`: Enables Ray's built-in autoscaling feature.
   - `autoscalerOptions`: Configures autoscaler behavior, including idle timeout and resource allocation.

3. **Head Node Specification**:
   - Defines the Ray head node configuration, including CPU and memory resources.
   - Exposes necessary ports for GCS, dashboard, and client communication.

4. **Worker Group Specification**:
   - Defines the worker nodes configuration.
   - Sets up autoscaling parameters:
     - `replicas: 0`: Initial number of worker nodes.
     - `minReplicas: 0`: Minimum number of worker nodes.
     - `maxReplicas: 20`: Maximum number of worker nodes the cluster can scale to.

The autoscaler will automatically adjust the number of worker nodes based on the cluster's workload, scaling up when more resources are needed and scaling down during idle periods.

## ray-cluster-service-public.yaml

The `ray-cluster-service-public.yaml` file creates a Kubernetes Service that exposes the Ray cluster's head node to the public internet. Key aspects include:

This Service allows external clients to connect to the Ray cluster, access the dashboard for monitoring, and interact with the cluster's services.
