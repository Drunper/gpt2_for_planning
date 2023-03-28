import xml.etree.ElementTree as ET


class Action:
    def __init__(self, xml_action):
        root = ET.fromstring(xml_action)
        self.name = root.find("Name").text[1:-1]
        self.id = root.find("Id").text
        self.parent = root.find("Parent").text
        self.student = root.find("Student").text
        self.predecessor = [child.find("Id").text for child in root.find("Predecessors")]
        self.precondition = [child.text[1:-1] for child in root.find("Preconds")]
        self.positiveEffects = [child.text[1:-1] for child in root.find("PositiveEffects")]
        self.negativeEffects = [child.text[1:-1] for child in root.find("NegativeEffects")]
