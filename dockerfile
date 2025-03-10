# Define custom function directory
ARG FUNCTION_DIR="/function"

FROM python:3.12 AS build-image

# Include global arg in this stage of the build
ARG FUNCTION_DIR

# Copy function code
RUN mkdir -p ${FUNCTION_DIR}
COPY README.md pyproject.toml poetry.lock requirements.txt ${FUNCTION_DIR}/

# Install poetry
RUN pip install poetry

COPY . ${FUNCTION_DIR}/

# Set working directory
WORKDIR ${FUNCTION_DIR}

# Install the dependencies using poetry.
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --without dev

# Copy the rest of the files, including source code

# Install awslambdaric
RUN pip install --target ${FUNCTION_DIR} awslambdaric

# Use a slim version of the base Python image to reduce the final image size
#FROM python:3.12-slim

# Include global arg in this stage of the build
#ARG FUNCTION_DIR

# Set working directory to function root directory
#WORKDIR ${FUNCTION_DIR}

# Copy in the built dependencies
#COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "python", "-m", "awslambdaric" ]
# Pass the name of the function handler as an argument to the runtime
CMD [ "lambda_function.handler" ]
