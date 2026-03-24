# Security: Vector Operation communicates with upstream services via internal network.
# TLS termination is handled at the infrastructure level
# (e.g. Nginx reverse proxy, Kubernetes Ingress, or cloud load balancer).
# No application-level authentication is implemented in this service.
