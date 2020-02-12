from marshmallow import RAISE

from argschema import ArgSchema

from argschema.fields import (
    LogLevel,
    Dict,
    String,
    Int,
    DateTime,
    Nested,
    Boolean,
    Float,
)

from allensdk.brain_observatory.argschema_utilities import (
    check_read_access,
    check_write_access,
    RaisingSchema,
)

class InputSchema(ArgSchema):

    experiment_id = Int(required=True,
                        help='Brain Observatory experiment ID')

    manifest_path = String(required=True,
                           help='Path to Brain Observatory manifest.json file')

    temp_directory = String(required=True,
                            #validate=check_write_access,
                            help='Path to temporary directory for storing intermediate files')

    output_path = String(required=True,
                             #validate=check_write_access,
                            help='Path to NWB 2.0 file to be generated')

class OutputSchema(RaisingSchema):
    nwb_path = String(required=True, description='path to output file')


# MODULE TODO:
# - update metadata for ophys experiments
# - extract raw running wheel data from original NWB file
