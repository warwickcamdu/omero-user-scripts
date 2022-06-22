# import the omero package and the omero.scripts package.
import omero
import omero.scripts as scripts
from omero.gateway import BlitzGateway, DatasetWrapper
from omero.rtypes import rlong, rstring, robject
import omero.util.script_utils as script_utils
import numpy as np
'''
Slow, low memory usage
'''


def log(data):
    """Handle logging or printing in one place."""
    print(data)

def getImages(conn, script_params):
    """
    Get the images
    """
    message = ""
    objects, log_message = script_utils.get_objects(conn, script_params)
    message += log_message
    if not objects:
        return None, message

    data_type = script_params["Data_Type"]

    if data_type == 'Dataset':
        images = []
        for ds in objects:
            images.extend(list(ds.listChildren()))
        if not images:
            message += "No image found in dataset(s)"
            return None, message
    else:
        images = objects
    return images


def get_plane(raw_pixel_store, pixels, the_z, the_c, the_t):
    # get the plane
    pixels_id = pixels.getId().getValue()
    raw_pixel_store.setPixelsId(pixels_id, True)
    return script_utils.download_plane(raw_pixel_store, pixels, the_z, the_c, the_t)

    
def stdCalculator(channel,raw_pixel_store, pixels, sizeZ):
    """
    Caluclate standard deviation of plane
    """
    max_std = 0
    for z in range(sizeZ):
        plane = get_plane(raw_pixel_store, pixels, z, channel, 0)
        std = np.std(plane)
        if std > max_std:
            max_std = std
            slice = z
    return slice + 1

def runScript():
    dataTypes = [rstring('Dataset'), rstring('Image')]
    client = scripts.client(
        "Find_in_focus_plane.py", """Identify the most in focus plane 
        (from the first time step). Outputs Z plane number to be used with
        Batch Image Export script""",
        scripts.String(
            "Data_Type", optional=False, grouping="01", values=dataTypes,
            default="Image"),
        scripts.List(
            "IDs", optional=False, grouping="02",
            description="""IDs of the images to process""").ofType(rlong(0)),
        scripts.Int(
            "Channel", optional=False, grouping="03",
            description="""Channel to analyse""", min=0, default=0),
        version="0.0",
        authors=["Laura Cooper", "CAMDU"],
        institutions=["University of Warwick"],
        contact="camdu@warwick.ac.uk"
        )

    try:
        conn = BlitzGateway(client_obj=client)
        script_params = client.getInputs(unwrap=True)
        images = getImages(conn, script_params)
        # Create new dataset if Dataset_Name is defined
        for image in images:
            sizeZ=image.getSizeZ()
            # Skip image if Z dimension is 1 or if given Z range is less than 1
            if (sizeZ > 1):
                # Get plane as numpy array
                raw_pixel_store = conn.c.sf.createRawPixelsStore()
                query_service = conn.getQueryService()
                query_string = "select p from Pixels p join fetch p.image i "\
                    "join fetch p.pixelsType pt where i.id='%d'" % image.getId()
                pixels = query_service.findByQuery(query_string, None)
                z = stdCalculator(script_params["Channel"], raw_pixel_store, pixels, sizeZ)
                print("Image ID: ", image.getId(), " In focus plane: ", z)

    finally:
        # Cleanup
        client.closeSession()


if __name__ == '__main__':
    runScript()
