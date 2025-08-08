# import the omero package and the omero.scripts package.
import omero
import omero.scripts as scripts
from omero.gateway import BlitzGateway
from omero.rtypes import rlong, rstring, rdouble
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
    
def create_roi(image, x,y,width,height):
    """
    Creates a rectangle ROI
    :param image: The image
    :param point_str: The points as string (x1,y1, x2,y2, ...)
    :return: The ROI
    """
    roi = omero.model.RoiI()
    roi.setImage(image._obj)
    rect = omero.model.RectangleI()
    rect.x = rdouble(x)
    rect.y = rdouble(y)
    rect.width = rdouble(width)
    rect.height = rdouble(height)
    roi.addShape(rect)
    return roi

def runScript():
    dataTypes = [rstring('Image')]
    client = scripts.client(
        "Specify_ROI.py", """Enter coordinates for rectangular ROI, the ROI is added to the image""",
        scripts.String(
            "Data_Type", optional=False, grouping="01", values=dataTypes,
            default="Image"),
        scripts.List(
            "IDs", optional=False, grouping="02",
            description="""IDs of the images to process""").ofType(rlong(0)),
        scripts.Float(
            "X", optional=False, grouping="03", description="""x coordinate"""),
        scripts.Float(
            "Y", optional=False, grouping="04", description="""y coordinate"""),
        scripts.Float(
            "Width", optional=False, grouping="05", description="""width of the rectangle"""),
        scripts.Float(
            "Height", optional=False, grouping="06", description="""height of the rectangle"""),
        #scripts.Int(
        #    "Channel", optional=False, grouping="03",
        #    description="""Channel to analyse""", min=0, default=0),
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
            roi = create_roi(image, script_params['X'], script_params['Y'], script_params['Width'], script_params['Height'])
            conn.getUpdateService().saveAndReturnObject(roi)
    finally:
        # Cleanup
        client.closeSession()


if __name__ == '__main__':
    runScript()
