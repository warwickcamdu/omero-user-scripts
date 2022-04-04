# import the omero package and the omero.scripts package.
import omero
import omero.scripts as scripts
from omero.gateway import BlitzGateway, DatasetWrapper
from omero.rtypes import rlong, rstring, robject
import omero.util.script_utils as script_utils
import numpy as np
from time import time
'''
Slow, low memory usage
'''


def log(data):
    """Handle logging or printing in one place."""
    print(data)


def copyMetadata(conn, newImage, image):
    """
    Copy important metadata
    Reload to prevent update conflicts
    """
    newImage = conn.getObject("Image", newImage.getId())
    new_pixs = newImage.getPrimaryPixels()._obj
    old_pixs = image.getPrimaryPixels()._obj
    new_pixs.setPhysicalSizeX(old_pixs.getPhysicalSizeX())
    new_pixs.setPhysicalSizeY(old_pixs.getPhysicalSizeY())
    new_pixs.setPhysicalSizeZ(old_pixs.getPhysicalSizeZ())
    conn.getUpdateService().saveObject(new_pixs)
    for old_channels, new_channels in zip(image.getChannels(),
                                          newImage.getChannels()):
        new_LogicChan = new_channels._obj.getLogicalChannel()
        new_LogicChan.setName(rstring(old_channels.getLabel()))
        new_LogicChan.setEmissionWave(old_channels.getEmissionWave(units=True))
        new_LogicChan.setExcitationWave(
            old_channels.getExcitationWave(units=True))
        conn.getUpdateService().saveObject(new_LogicChan)

    if newImage._prepareRenderingEngine():
        newImage._re.resetDefaultSettings(True)


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


def getRoiShape(s):
    shape = {}
    shape['x'] = int(np.floor(s.getX().getValue()))
    shape['y'] = int(np.floor(s.getY().getValue()))
    shape['w'] = int(np.floor(s.getWidth().getValue()))
    shape['h'] = int(np.floor(s.getHeight().getValue()))
    return shape


def get_plane(raw_pixel_store, pixels, the_z, the_c, the_t):
    # get the plane
    pixels_id = pixels.getId().getValue()
    raw_pixel_store.setPixelsId(pixels_id, True)
    return script_utils.download_plane(raw_pixel_store, pixels, the_z, the_c, the_t)


def planeGenerator(new_Z, C, T, Z, raw_pixel_store, pixels, projection, shape=None):
    """
    Set up generator of 2D numpy arrays, each of which is a MIP
    To be passed to createImage method so must be order z, c, t
    """
    for z in range(new_Z):  # createImageFromNumpySeq expects Z, C, T order
        for c in range(C):
            for t in range(T[0]-1, T[1]):
                for eachz in range(Z[0]-1, Z[1]):
                    #plane = pixels.getPlane(eachz, c, t)
                    plane = get_plane(raw_pixel_store, pixels, 0, 0, 0)
                    if shape is not None:
                        plane = plane[shape['y']:shape['y']+shape['h'],
                                      shape['x']:shape['x']+shape['w']]
                    if eachz == Z[0]-1:
                        new_plane = plane
                    else:
                        if projection == 'Maximum':
                            # Replace pixel values if larger
                            new_plane = np.where(np.greater(
                                plane, new_plane), plane, new_plane)
                        elif projection == 'Minimum':
                            new_plane = np.where(
                                np.less(plane, new_plane), plane, new_plane)
                yield new_plane


def create_new_dataset(conn, name):
    new_dataset = DatasetWrapper(conn, omero.model.DatasetI())
    new_dataset.setName(name)
    new_dataset.save()
    return new_dataset


def runScript():
    dataTypes = [rstring('Dataset'), rstring('Image')]
    projections = [rstring('Maximum'), rstring('Minimum')]
    client = scripts.client(
        "Intensity_Projection.py", """Creates a new image of the selected \
        intensity projection in Z from an existing image""",
        scripts.String(
            "Data_Type", optional=False, grouping="01", values=dataTypes,
            default="Image"),
        scripts.List(
            "IDs", optional=False, grouping="02",
            description="""IDs of the images to project""").ofType(rlong(0)),
        scripts.String(
            "Method", grouping="03",
            description="""Type of projection to run""", values=projections,
            default='Maximum'),
        scripts.Int(
            "First_Z", grouping="04", min=1,
            description="First Z plane to project, default is first plane"),
        scripts.Int(
            "Last_Z", grouping="05", min=1,
            description="Last Z plane to project, default is last plane"),
        scripts.Int(
            "First_T", grouping="06", min=1,
            description="First T plane to project, default is first plane"),
        scripts.Int(
            "Last_T", grouping="07", min=1,
            description="Last T plane to project, default is last plane"),
        scripts.Bool(
            "Apply_to_ROIs_only", grouping="08", default=False,
            description="Apply maximum projection only to rectangular ROIs, \
            if not rectangular ROIs found, image will be skipped"),
        scripts.String(
            "Dataset_Name", grouping="09",
            description="To save projections to new dataset, enter it's name. \
            To save projections to existing dataset, leave blank"),

        version="0.4",
        authors=["Laura Cooper", "CAMDU"],
        institutions=["University of Warwick"],
        contact="camdu@warwick.ac.uk"
        )

    try:
        for j in range(20):
            start_time = time()
            conn = BlitzGateway(client_obj=client)
            script_params = client.getInputs(unwrap=True)
            images = getImages(conn, script_params)
            user = conn.getUser()

            # Create new dataset if Dataset_Name is defined
            if "Dataset_Name" in script_params:
                new_dataset = create_new_dataset(conn,
                                                 script_params["Dataset_Name"])

            for image in images:
                # If Dataset_Name empty user existing, use new one if not.
                if "Dataset_Name" in script_params:
                    dataset = new_dataset
                else:
                    dataset = image.getParent()
                    if dataset.getOwnerOmeName() != user:
                        dataset = create_new_dataset(conn, dataset.getName())
                Z, C, T = image.getSizeZ(), image.getSizeC(), image.getSizeT()
                if "First_Z" in script_params:
                    Z1 = [script_params["First_Z"], Z]
                else:
                    Z1 = [1, Z]
                if "Last_Z" in script_params:
                    Z1[1] = script_params["Last_Z"]
                if "First_T" in script_params:
                    T1 = [script_params["First_T"], T]
                else:
                    T1 = [1, T]
                if "Last_T" in script_params:
                    T1[1] = script_params["Last_T"]
                # Skip image if Z dimension is 1 or if given Z range is less than 1
                if (Z != 1) or ((Z1[1]-Z1[0]) >= 1):
                    # Get plane as numpy array
                    raw_pixel_store = conn.c.sf.createRawPixelsStore()
                    query_service = conn.getQueryService()
                    query_string = "select p from Pixels p join fetch p.image i join fetch p.pixelsType pt where i.id='%d'" % int(
                        script_params["IDs"][0])
                    pixels = query_service.findByQuery(query_string, None)
                    if script_params["Apply_to_ROIs_only"]:
                        roi_service = conn.getRoiService()
                        result = roi_service.findByImage(image.getId(), None)
                        if result is not None:
                            for roi in result.rois:
                                for s in roi.copyShapes():
                                    if type(s) == omero.model.RectangleI:
                                        shape = getRoiShape(s)
                                        name = "%s_%s_%s" % (image.getName(),
                                                             s.getId().getValue(),
                                                             script_params["Method"])
                                        desc = ("%s intensity Z projection of\
                                                Image ID: %s, shape ID: %s"
                                                % (script_params["Method"],
                                                   image.getId(),
                                                   s.getId().getValue()))
                    else:
                        shape = {}
                        shape['x'] = 0
                        shape['y'] = 0
                        shape['w'] = image.getSizeX()
                        shape['h'] = image.getSizeY()
                        name = "%s_%s" % (
                            image.getName(), script_params["Method"])
                        desc = ("%s intensity Z projection of Image ID: \
                                 %s" % (script_params["Method"],
                                        image.getId()))
                    newImage = conn.createImageFromNumpySeq(
                        planeGenerator(1, C, T1, Z1, raw_pixel_store, pixels,
                                       script_params["Method"], shape),
                        name, 1, C, T1[1]-T1[0], description=desc, dataset=dataset)
                    copyMetadata(conn, newImage, image)
                    client.setOutput("New Image", robject(newImage._obj))
            end_time = time()
            print("run time:", end_time - start_time)

    finally:
        # Cleanup
        client.closeSession()



if __name__ == '__main__':
    runScript()
