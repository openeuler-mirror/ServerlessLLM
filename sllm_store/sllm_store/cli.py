# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #

import asyncio
import sys
import logging
import click

from sllm_store.server import serve
from sllm_store.logger import init_logger
from sllm_store.utils import to_num_bytes

# Initialize logger for this module
logger = init_logger(__name__)


@click.group()
def cli():
    """sllm-store CLI"""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host for the gRPC server")
@click.option("--port", default=8073, help="Port for the gRPC server")
@click.option("--storage-path", default="./models", help="Local path to store models")
@click.option("--num-thread", default=4, help="Number of I/O threads for the server")
@click.option(
    "--chunk-size", default="32MB", help="Chunk size for data transfer, e.g., 4KB, 1MB, 1GB"
)
@click.option(
    "--mem-pool-size",
    default="4GB",
    help="Memory pool size for the server, e.g., 1GB, 4GB, 1TB",
)
@click.option(
    "--disk-size", default="128GB", help="Total disk size available, e.g., 1GB, 4GB, 1TB (Note: currently not used in 'serve' function)"
)
@click.option(
    "--registration-required",
    is_flag=True,  # Changed to is_flag for boolean option without explicit value
    default=False,
    help="Require registration before loading model (set this flag to enable)",
)
def start(
    host,
    port,
    storage_path,
    num_thread,
    chunk_size,
    mem_pool_size,
    disk_size, # This parameter is passed but commented out in the serve call below.
    registration_required,
):
    """Start the gRPC server for ServerlessLLM Store"""
    # Convert the chunk size string to bytes (e.g., "32MB" -> 33554432)
    chunk_size = to_num_bytes(chunk_size)

    # Convert the memory pool size string to bytes
    mem_pool_size = to_num_bytes(mem_pool_size)

    try:
        logger.info(f"Starting gRPC server on {host}:{port}")
        # Run the asynchronous gRPC server
        asyncio.run(
            serve(
                host=host,
                port=port,
                storage_path=storage_path,
                num_thread=num_thread,
                chunk_size=chunk_size,
                mem_pool_size=mem_pool_size,
                # disk_size is not currently used by the serve function, as indicated in the original code.
                # disk_size=disk_size,
                registration_required=registration_required,
            )
        )
    except KeyboardInterrupt:
        logging.info("Server stopped by user (KeyboardInterrupt)")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


# Entry point for the 'sllm-store' command line tool
def main():
    """Main function to run the CLI"""
    cli()

if __name__ == "__main__":
    main()
