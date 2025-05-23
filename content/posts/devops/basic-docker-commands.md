+++
categories = ['DevOps']
tags = ['Tutorials', 'Docker']
title = 'Basic Docker Commands'
keywords = ['Docker installation on Linux', 'Install Docker on Ubuntu', 'Install Docker on CentOS', 'Docker installation steps for Linux', 'Linux Docker setup', 'Docker on Linux distributions', 'Ubuntu Docker installation guide', 'CentOS Docker installation guide', 'Linux Docker configuration', 'Docker repository setup on Linux', 'Docker engine installation on Linux',
'Docker installation on Windows', 'Install Docker Desktop on Windows', 'Docker on Windows 10', 'Docker on Windows 11', 'Windows Docker setup', 'Docker Hyper-V and WSL 2 setup', 'Docker Desktop installation on Windows', 'Windows Docker configuration', 'Docker with WSL 2 on Windows', 'Docker without Docker Desktop on Windows',
'Docker installation on macOS', 'Install Docker Desktop on macOS', 'Docker on Mac with Apple silicon', 'macOS Docker setup', 'Docker configuration on macOS', 'Install Docker using Rosetta 2 on macOS', 'Docker Desktop for Mac installation', 'Verify Docker installation on macOS', 'macOS Docker troubleshooting',
'Docker installation guide', 'How to install Docker', 'Docker setup and configuration', 'Docker for developers', 'Containerization with Docker', 'Docker engine installation', 'Docker CLI commands', 'Docker environment setup', 'Docker best practices',
'Step-by-step Docker installation on Linux', 'Troubleshooting Docker installation on Windows', 'Optimizing Docker performance on macOS', 'Docker installation requirements for Linux', 'Docker Desktop features on Windows', 'Docker security best practices on macOS', 'Docker networking setup on Linux', 'Docker volumes and persistent storage on Windows', 'Docker Compose installation on macOS']
hide_site_title = true
og_type = 'article'
date = 2024-12-20T15:47:32+05:30
[cover]
    image = 'https://images.unsplash.com/photo-1524741978410-350ba91a70d7?q=80&w=1528&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
+++

Docker and Docker Compose are powerful tools for containerization and orchestration, simplifying the process of developing, deploying, and managing applications. Here is a detailed guide to the various commands you will need to manage your Docker and Docker Compose environments.

### Docker Management Commands

These commands are essential for managing and understanding your Docker setup.

#### General Commands

- **`docker --help`**:
  - Displays help for the Docker CLI and its subcommands. This is a great starting point if you need to understand the available commands and their options.
  - Example: `docker --help`

- **`docker -d`**:
  - Starts the Docker daemon. This command is typically used when you need to start the Docker service manually.
  - Example: `docker -d`

- **`docker info`**:
  - Displays system-wide information about Docker, including the number of containers, images, and other system details.
  - Example: `docker info`

- **`docker version`**:
  - Displays the version of Docker installed on your system.
  - Example: `docker version`

### Docker Hub Commands

Docker Hub is a central registry for Docker images, allowing you to push, pull, and manage images.

#### Image Management

- **`docker login`**:
  - Logs in to Docker Hub or another Docker registry. You need to specify your username.
  - Example: `docker login -u <username>`

- **`docker search`**:
  - Searches for Docker images on Docker Hub. This command helps you find images based on keywords.
  - Example: `docker search <image_name>`

- **`docker pull`**:
  - Pulls a Docker image from Docker Hub or another registry. This command downloads the specified image to your local machine.
  - Example: `docker pull <username>/<image_name>`

- **`docker push`**:
  - Pushes a Docker image to Docker Hub or another registry. This command uploads the specified image to the registry.
  - Example: `docker push <username>/<image_name>`

- **`docker save`**:
  - Saves a Docker image to a tar archive. This is useful for backing up or transferring images without using a registry.
  - Example: `docker save <image_name> > image.tar`

- **`docker load`**:
  - Loads a Docker image from a tar archive. This command is the counterpart to `docker save`.
  - Example: `docker load < image.tar`

### Docker Image Commands

Managing Docker images is crucial for maintaining your containerized applications.

#### Image Operations

- **`docker images`**:
  - Lists all available Docker images on the system. This command provides a list of all images, including their IDs, tags, and sizes.
  - Example: `docker images`

- **`docker rmi`**:
  - Removes one or more Docker images. This command helps in cleaning up unused images.
  - Example: `docker rmi <image_name>`

- **`docker build`**:
  - Builds a Docker image from a Dockerfile. This command is used to create a new image based on the instructions in the Dockerfile.
  - Example: `docker build -t <image_name> .`

- **`docker inspect`**:
  - Inspects a Docker image for detailed information. This command provides detailed metadata about the image.
  - Example: `docker inspect <image_name>`

- **`docker tag`**:
  - Creates a tag for a Docker image. This is useful for creating aliases or versions of an image.
  - Example: `docker tag <image_name> <new_tag>`

### Docker Container Commands

Managing containers is a core part of using Docker.

#### Container Creation and Management

- **`docker container create`**:
  - Creates a new container but does not start it.
  - Example: `docker container create --name my_container <image_name>`

- **`docker container run`**:
  - Creates and starts a new container from an image. This command has several options:
    - **`--name`**: Specifies the name of the container.
    - **`-p`**: Maps ports between the host and container.
    - **`-d`**: Runs the container in detached mode (background).
    - **`--rm`**: Automatically removes the container when it exits.
  - Example: `docker run --name my_container -p 8080:80 -d <image_name>`

- **`docker container start`**:
  - Starts a stopped container.
  - Example: `docker container start <container_name>`

- **`docker container stop`**:
  - Stops a running container.
  - Example: `docker container stop <container_name>`

- **`docker container restart`**:
  - Restarts a running container.
  - Example: `docker container restart <container_name>`

- **`docker container rm`**:
  - Removes one or more stopped containers.
  - Example: `docker container rm <container_name>`

#### Container Inspection and Debugging

- **`docker container ls`**:
  - Lists running containers. Options include:
    - **`-a`**: Lists all containers (running and stopped).
    - **`-l`**: Lists the latest created container.
    - **`-q`**: Lists only the container IDs.
  - Example: `docker container ls -a`

- **`docker container logs`**:
  - Displays logs from a container. Options include:
    - **`-f`**: Follows the log output.
    - **`--tail=N`** or **`--tail=all`**: Shows the last N logs or all logs.
  - Example: `docker container logs -f <container_name>`

- **`docker container exec`**:
  - Executes a command in a running container.
  - Example: `docker container exec -it <container_name> sh`

- **`docker container inspect`**:
  - Inspects a running container for detailed information.
  - Example: `docker container inspect <container_name>`

- **`docker container stats`**:
  - Displays resource usage statistics for containers.
  - Example: `docker container stats <container_name>`

- **`docker container top`**:
  - Displays the running processes of a container.
  - Example: `docker container top <container_name>`

- **`docker container wait`**:
  - Blocks until one or more containers stop, then prints their exit codes.
  - Example: `docker container wait <container_name>`

- **`docker container pause`**:
  - Pauses all processes within one or more containers.
  - Example: `docker container pause <container_name>`

- **`docker container unpause`**:
  - Unpauses all processes within one or more containers.
  - Example: `docker container unpause <container_name>`

- **`docker container update`**:
  - Updates the configuration of one or more containers.
  - Example: `docker container update --memory 512m <container_name>`

- **`docker container kill`**:
  - Kills one or more running containers.
  - Example: `docker container kill <container_name>`

- **`docker container port`**:
  - Lists port mappings or a specific mapping for the container.
  - Example: `docker container port <container_name> 80`

- **`docker container prune`**:
  - Removes all stopped containers.
  - Example: `docker container prune`

- **`docker container rename`**:
  - Renames a container.
  - Example: `docker container rename <old_name> <new_name>`

- **`docker container export`**:
  - Exports a container’s filesystem as a tar archive.
  - Example: `docker container export <container_name> > container.tar`

- **`docker container cp`**:
  - Copies files/folders between a container and the local filesystem.
  - Example: `docker container cp <container_name>:<path_in_container> <local_path>`

- **`docker container diff`**:
  - Inspects changes to files or directories on a container’s filesystem.
  - Example: `docker container diff <container_name>`

#### Rare but Useful Commands

- **`docker container commit`**:
  - Creates a new image from a container’s changes.
  - Example: `docker container commit <container_name> <new_image_name>`

### Docker Network Commands

Managing networks is essential for communication between containers.

#### Network Management

- **`docker network create`**:
  - Creates a new network.
  - Example: `docker network create my_network`

- **`docker network connect`**:
  - Connects a container to a network.
  - Example: `docker network connect my_network <container_name>`

- **`docker network disconnect`**:
  - Disconnects a container from a network.
  - Example: `docker network disconnect my_network <container_name>`

- **`docker network rm`**:
  - Removes one or more networks.
  - Example: `docker network rm my_network`

- **`docker network ls`**:
  - Lists all networks.
  - Example: `docker network ls`

### Docker Volume Commands

Managing volumes helps in persisting data across container restarts.

#### Volume Management

- **`docker volume create`**:
  - Creates a new volume.
  - Example: `docker volume create my_volume`

- **`docker volume inspect`**:
  - Inspects a volume for detailed information.
  - Example: `docker volume inspect my_volume`

- **`docker volume ls`**:
  - Lists all volumes.
  - Example: `docker volume ls`

- **`docker volume rm`**:
  - Removes one or more volumes.
  - Example: `docker volume rm my_volume`

### Docker Compose Commands

Docker Compose simplifies the process of managing multi-container applications.

#### General Compose Commands

- **`docker compose --help`**:
  - Displays help for Docker Compose and its subcommands.
  - Example: `docker compose --help`

#### Service Management

- **`docker compose build`**:
  - Builds or rebuilds services defined in the Compose file.
  - Example: `docker compose build`

- **`docker compose config`**:
  - Validates and displays the Compose file configuration.
  - Example: `docker compose config`

- **`docker compose create`**:
  - Creates containers for services defined in the Compose file.
  - Example: `docker compose create`

- **`docker compose down`**:
  - Stops and removes containers, networks, and volumes defined in the Compose file. Options include:
    - **`-v`**: Removes named volumes.
    - **`--rmi all`**: Removes all images used by the services.
  - Example: `docker compose down -v --rmi all`

- **`docker compose events`**:
  - Receives real-time events from containers.
  - Example: `docker compose events`

- **`docker compose exec`**:
  - Executes a command in a running container defined in the Compose file.
  - Example: `docker compose exec <service_name> sh`

- **`docker compose images`**:
  - Lists images used by the created containers.
  - Example: `docker compose images`

- **`docker compose kill`**:
  - Force stops service containers.
  - Example: `docker compose kill`

- **`docker compose logs`**:
  - Displays logs from containers defined in the Compose file. Options include:
    - **`-f`**: Follows the log output.
    - **`--tail=N`** or **`--tail=all`**: Shows the last N logs or all logs.
  - Example: `docker compose logs -f`

- **`docker compose ls`**:
  - Lists running Compose projects.
  - Example: `docker compose ls`

- **`docker compose pause`**:
  - Pauses services defined in the Compose file.
  - Example: `docker compose pause`

- **`docker compose port`**:
  - Prints the public port for a port binding.
  - Example: `docker compose port <service_name> <port>`

- **`docker compose ps`**:
  - Lists containers defined in the Compose file. Options include:
    - **`-q`**: Limits the display to container IDs.
    - **`-a`**: Shows all stopped containers.
  - Example: `docker compose ps -a`

- **`docker compose pull`**:
  - Pulls service images defined in the Compose file.
  - Example: `docker compose pull`

- **`docker compose push`**:
  - Pushes service images defined in the Compose file.
  - Example: `docker compose push`

- **`docker compose restart`**:
  - Restarts service containers defined in the Compose file.
  - Example: `docker compose restart`

- **`docker compose rm`**:
  - Removes stopped service containers defined in the Compose file.
  - Example: `docker compose rm`

- **`docker compose run`**:
  - Runs a one-off command on a service defined in the Compose file.
  - Example: `docker compose run <service_name> sh`

- **`docker compose start`**:
  - Starts services defined in the Compose file.
  - Example: `docker compose start`

- **`docker compose stop`**:
  - Stops services defined in the Compose file.
  - Example: `docker compose stop`

- **`docker compose top`**:
  - Displays the running processes in services defined in the Compose file.
  - Example: `docker compose top`

- **`docker compose unpause`**:
  - Unpauses services defined in the Compose file.
  - Example: `docker compose unpause`

- **`docker compose up`**:
  - Builds, (re)creates, starts, and attaches to containers for services defined in the Compose file. Options include:
    - **`-d`**: Starts containers in detached mode.
    - **`--scale`**: Scales the number of services.
  - Example: `docker compose up -d --scale <service_name>=3`

- **`docker compose version`**:
  - Displays the Docker Compose version information.
  - Example: `docker compose version`

- **`docker compose wait`**:
  - Blocks until containers of all (or specified) services stop.
  - Example: `docker compose wait`

- **`docker compose watch`**:
  - Watches the build context for services and rebuilds/refreshes containers when files are updated. Note that this command is not officially supported and may require additional tools.
  - Example: This command is not standard and may vary based on the tool used.

#### Specifying Multiple Compose Files

You can supply multiple Compose files to combine configurations:

- **Using the `-f` flag**:
  - Specifies the location of one or more Compose configuration files.
  - Example: `docker compose -f docker-compose.yml -f docker-compose.admin.yml run backup_db`

- **Using `stdin`**:
  - Reads the configuration from `stdin`.
  - Example: `docker compose -f - < docker-compose.yml`

#### Specifying a Project Name

- **Using the `-p` flag**:
  - Specifies an alternate project name.
  - Example: `docker compose -p my_project up`

### Docker Swarm Commands

Docker Swarm is used for container orchestration at scale.

#### Swarm Management

- **`docker node ls`**:
  - Lists nodes in the swarm.
  - Example: `docker node ls`

- **`docker service create`**:
  - Creates a new service in the swarm.
  - Example: `docker service create --name my_service <image_name>`

- **`docker service ls`**:
  - Lists services in the swarm.
  - Example: `docker service ls`

- **`docker service scale`**:
  - Scales services in the swarm.
  - Example: `docker service scale my_service=3`

- **`docker service rm`**:
  - Removes a service from the swarm.
  - Example: `docker service rm my_service`

## Conclusion

Understanding and mastering these Docker and Docker Compose commands is crucial for efficiently managing containerized applications. From basic management to advanced orchestration, these commands cover a wide range of tasks, ensuring you have the tools needed to manage your Docker environment effectively.