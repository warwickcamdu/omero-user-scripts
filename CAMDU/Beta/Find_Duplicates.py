# import the omero package and the omero.scripts package.
import omero
import omero.scripts as scripts
from omero.gateway import BlitzGateway
from omero.rtypes import rlong, rstring
import pandas as pd
'''
Find duplicate images within same dataset and move to project for deletion
'''


def log(data):
    """Handle logging or printing in one place."""
    print(data)


def runScript():
    data_types = [rstring('Dataset')]
    client = scripts.client(
        "Find_Duplicates.py",
        """
        Find duplicate images within a dataset and tag with "CAMDU Duplicate"
        so admin can delete them
        """,
        scripts.String(
            "Data_Type", optional=False, grouping="1",
            description="Choose Dataset", values=data_types,
            default="Dataset"),
        scripts.List(
            "IDs", optional=False, grouping="2",
            description="List of Dataset IDs to process.").ofType(rlong(0)),
        version="0.0",
        authors=["Laura Cooper", "CAMDU"],
        institutions=["University of Warwick"],
        contact="camdu@warwick.ac.uk"
        )
    try:
        conn = BlitzGateway(client_obj=client)
        script_params = client.getInputs(unwrap=True)
        roi_service = conn.getRoiService()

        for id in script_params["IDs"]:
            dataset = conn.getObject("Dataset", id)
            colNames = ['id', 'Name', 'acDate', 'sizeX', 'sizeY', 'sizeZ',
                        'sizeT', 'sizeC', 'No. Annotate', 'No. ROI']
            metadata = pd.DataFrame(columns=colNames)
            for image in dataset.listChildren():
                findRois = roi_service.findByImage(image.getId(), None)
                roiIds = [roi.getId().getValue() for roi in findRois.rois]
                # Get custom comments annotations
                # (ignore autogenerated at import using regex)
                anns = []
                for ann in image.listAnnotations():
                    if isinstance(ann, omero.gateway.CommentAnnotationWrapper):
                        if not ann.getTextValue().startswith('regex'):
                            anns.append(ann)
                    else:
                        anns.append(ann)

                image_data = pd.DataFrame(data={'id': image.getId(),
                                                'Name': image.getName(),
                                                'acDate': image.getDate(),
                                                'sizeX': image.getSizeX(),
                                                'sizeY': image.getSizeY(),
                                                'sizeZ': image.getSizeZ(),
                                                'sizeT': image.getSizeT(),
                                                'sizeC': image.getSizeC(),
                                                'No. Annotate': len(anns),
                                                'No. ROI':  len(roiIds)
                                                }, index=[0])
                metadata = metadata.append(image_data)
            # Remove unique acquisition dates
            mask = metadata.duplicated(subset=colNames[1::], keep='first')
            if not metadata[mask].empty:
                tag_ann = omero.gateway.TagAnnotationWrapper(conn)
                tag_ann.setValue("CAMDU Duplicate")
                tag_ann.setDescription(
                    "Duplicate image to be deleted by CAMDU")
                tag_ann.save()
                for id in metadata[mask]['id']:
                    image = conn.getObject("Image", id)
                    image.linkAnnotation(tag_ann)
    finally:
        # Cleanup
        client.closeSession()


if __name__ == '__main__':
    runScript()
