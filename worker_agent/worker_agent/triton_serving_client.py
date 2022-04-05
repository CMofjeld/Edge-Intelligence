"""Class providing an interface to NVIDIA Triton Inference Server.

The implementation for this class is primarily based on the code
provided in the Triton image client sample located at:
https://github.com/triton-inference-server/client/blob/main/src/python/examples/image_client.py
The original copyright notice is reproduced below, as required.
"""
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import io
from typing import Dict

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from attrdict import AttrDict
from PIL import Image
from tritonclient.utils import InferenceServerException, triton_to_np_dtype

from worker_agent.serving_client import ServingClient


class TritonImageClient(ServingClient):
    """Client for performing inference with Triton Inference Server."""

    def __init__(self, serving_url: str, protocol: str) -> None:
        # Instantiate client object from tritonclient library
        protocol = protocol.lower()
        if protocol == "grpc":
            self.triton_client = grpcclient.InferenceServerClient(url=serving_url)
        elif protocol == "http":
            self.triton_client = httpclient.InferenceServerClient(url=serving_url)
        else:
            raise InferenceServerException(
                f"invalid protocol for TritonImageClient: {protocol}"
            )
        self.protocol = protocol

        # Initialize cache of model metadata
        self.model_info = dict()

    def predict(self, image: bytes, model_id: str) -> Dict:
        """Send the given image for inference using the specified model and return the results.

        Args:
            image (bytes): binary image data
            model_id (str): ID of the model to perform inference with

        Returns:
            Dict: JSON result returned by serving software
        """
        # Get the metadata and configuration for the model
        if model_id not in self.model_info:
            # Request model metadata and config from the server
            model_metadata = self.triton_client.get_model_metadata(model_name=model_id)
            model_config = self.triton_client.get_model_config(model_name=model_id)
            if self.protocol == "grpc":
                model_config = model_config.config
            else:
                model_metadata, model_config = self._convert_http_metadata_config(
                    model_metadata, model_config
                )

            # Parse the response
            self.model_info[model_id] = self._parse_model(model_metadata, model_config)
        (
            max_batch_size,
            input_name,
            output_name,
            c,
            h,
            w,
            format,
            dtype,
        ) = self.model_info[model_id]

        # Preprocess the images into input data according to model
        # requirements
        image_arr = Image.open(io.BytesIO(image), formats=["JPEG"])
        image_data = [
            self._preprocess(image_arr, format, dtype, c, h, w, "NONE", self.protocol)
        ]
        if max_batch_size > 0:
            batched_image_data = np.stack(image_data, axis=0)
        else:
            batched_image_data = image_data[0]

        # Generate request
        if self.protocol == "grpc":
            client = grpcclient
        else:
            client = httpclient
        inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
        inputs[0].set_data_from_numpy(batched_image_data)

        # Send request
        response = self.triton_client.infer(model_name=model_id, inputs=inputs)

        # Return results
        return response.as_numpy(output_name).tolist()

    def _parse_model(self, model_metadata, model_config):
        """
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        if len(model_metadata.inputs) != 1:
            raise Exception(
                "expecting 1 input, got {}".format(len(model_metadata.inputs))
            )
        if len(model_metadata.outputs) != 1:
            raise Exception(
                "expecting 1 output, got {}".format(len(model_metadata.outputs))
            )

        if len(model_config.input) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(model_config.input)
                )
            )

        input_metadata = model_metadata.inputs[0]
        input_config = model_config.input[0]
        output_metadata = model_metadata.outputs[0]

        # Output is expected to be a vector. But allow any number of
        # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
        # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
        # is one.
        output_batch_dim = model_config.max_batch_size > 0
        non_one_cnt = 0
        for dim in output_metadata.shape:
            if output_batch_dim:
                output_batch_dim = False
            elif dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")

        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension), either CHW or HWC
        input_batch_dim = model_config.max_batch_size > 0
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".format(
                    expected_input_dims, model_metadata.name, len(input_metadata.shape)
                )
            )

        if type(input_config.format) == str:
            FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
            input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

        if (
            (input_config.format != mc.ModelInput.FORMAT_NCHW)
            and (input_config.format != mc.ModelInput.FORMAT_NHWC)
            and (input_config.format != mc.ModelInput.FORMAT_NONE)
        ):
            raise Exception(
                "unexpected input format "
                + mc.ModelInput.Format.Name(input_config.format)
                + ", expecting "
                + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW)
                + " or "
                + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC)
            )

        if (
            input_config.format == mc.ModelInput.FORMAT_NHWC
            or input_config.format == mc.ModelInput.FORMAT_NONE
        ):
            h = input_metadata.shape[1 if input_batch_dim else 0]
            w = input_metadata.shape[2 if input_batch_dim else 1]
            c = input_metadata.shape[3 if input_batch_dim else 2]
        else:
            c = input_metadata.shape[1 if input_batch_dim else 0]
            h = input_metadata.shape[2 if input_batch_dim else 1]
            w = input_metadata.shape[3 if input_batch_dim else 2]

        return (
            model_config.max_batch_size,
            input_metadata.name,
            output_metadata.name,
            c,
            h,
            w,
            input_config.format,
            input_metadata.datatype,
        )

    def _preprocess(self, img, format, dtype, c, h, w, scaling, protocol):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        """
        # np.set_printoptions(threshold='nan')

        if c == 1:
            sample_img = img.convert("L")
        else:
            sample_img = img.convert("RGB")

        resized_img = sample_img.resize((w, h), Image.BILINEAR)
        resized = np.array(resized_img)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        npdtype = triton_to_np_dtype(dtype)
        typed = resized.astype(npdtype)

        if scaling == "INCEPTION":
            scaled = (typed / 127.5) - 1
        elif scaling == "VGG":
            if c == 1:
                scaled = typed - np.asarray((128,), dtype=npdtype)
            else:
                scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
        else:
            scaled = typed

        # Swap to CHW if necessary
        if format == mc.ModelInput.FORMAT_NCHW:
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled

        # Channels are in RGB order. Currently model configuration data
        # doesn't provide any information as to other channel orderings
        # (like BGR) so we just assume RGB.
        return ordered

    def _convert_http_metadata_config(self, _metadata, _config):
        _model_metadata = AttrDict(_metadata)
        _model_config = AttrDict(_config)

        return _model_metadata, _model_config
