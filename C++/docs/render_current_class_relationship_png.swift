import AppKit

let width: CGFloat = 3508
let height: CGFloat = 2480
let boxWidth: CGFloat = 420
let boxHeight: CGFloat = 150
let headerHeight: CGFloat = 48

struct Box {
    let title: String
    let lines: [String]
    let x: CGFloat
    let yTop: CGFloat
    let headerColor: NSColor
    let bodyColor: NSColor

    var rect: NSRect {
        NSRect(x: x, y: height - yTop - boxHeight, width: boxWidth, height: boxHeight)
    }

    var center: NSPoint {
        NSPoint(x: rect.midX, y: rect.midY)
    }
}

func color(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat) -> NSColor {
    NSColor(calibratedRed: r / 255.0, green: g / 255.0, blue: b / 255.0, alpha: 1.0)
}

let border = color(33, 53, 71)
let arrowColor = color(49, 78, 82)
let titleColor = color(16, 32, 39)
let textColor = color(38, 50, 56)
let noteColor = color(124, 45, 18)

let boxes: [String: Box] = [
    "Cell": Box(title: "Cell", lines: ["abstract interface", "draw()", "drawOutline()"], x: 110, yTop: 1510, headerColor: color(255, 200, 87), bodyColor: color(255, 253, 248)),
    "Bacilli": Box(title: "Bacilli", lines: ["legacy subtype", "inherits Cell"], x: 110, yTop: 1790, headerColor: color(255, 217, 168), bodyColor: color(255, 253, 248)),
    "Sphere": Box(title: "Sphere", lines: ["standalone model", "uses SphereParams"], x: 600, yTop: 1510, headerColor: color(244, 162, 97), bodyColor: color(255, 253, 248)),
    "Spheroid": Box(title: "Spheroid", lines: ["primary tracked cell", "Frame stores vector<Spheroid>"], x: 600, yTop: 1790, headerColor: color(231, 111, 81), bodyColor: color(255, 253, 248)),
    "CellParams": Box(title: "CellParams", lines: ["base parameter record", "name"], x: 1110, yTop: 1280, headerColor: color(189, 224, 254), bodyColor: color(255, 253, 248)),
    "SphereParams": Box(title: "SphereParams", lines: ["x, y, z", "radius"], x: 1110, yTop: 1560, headerColor: color(207, 232, 255), bodyColor: color(255, 253, 248)),
    "SpheroidParams": Box(title: "SpheroidParams", lines: ["x, y, z", "major/minor radius", "theta_x/y/z"], x: 1110, yTop: 1980, headerColor: color(207, 232, 255), bodyColor: color(255, 253, 248)),
    "CellConfig": Box(title: "CellConfig", lines: ["abstract config base", "explodeConfig()"], x: 1540, yTop: 200, headerColor: color(144, 224, 239), bodyColor: color(255, 253, 248)),
    "SphereConfig": Box(title: "SphereConfig", lines: ["x, y, z, radius", "PerturbParams"], x: 1540, yTop: 500, headerColor: color(173, 232, 244), bodyColor: color(255, 253, 248)),
    "SpheroidConfig": Box(title: "SpheroidConfig", lines: ["x, y, z", "radii + theta params"], x: 1540, yTop: 820, headerColor: color(173, 232, 244), bodyColor: color(255, 253, 248)),
    "PerturbParams": Box(title: "PerturbParams", lines: ["prob", "mu", "sigma"], x: 1540, yTop: 1140, headerColor: color(202, 240, 248), bodyColor: color(255, 253, 248)),
    "SimulationConfig": Box(title: "SimulationConfig", lines: ["iterations_per_cell", "z_scaling", "z_slices"], x: 2080, yTop: 300, headerColor: color(149, 213, 178), bodyColor: color(255, 253, 248)),
    "ProbabilityConfig": Box(title: "ProbabilityConfig", lines: ["perturbation", "split", "split_cost"], x: 2080, yTop: 610, headerColor: color(149, 213, 178), bodyColor: color(255, 253, 248)),
    "BaseConfig": Box(title: "BaseConfig", lines: ["cellType", "cell*", "simulation", "prob"], x: 2080, yTop: 960, headerColor: color(116, 198, 157), bodyColor: color(255, 253, 248)),
    "CellFactory": Box(title: "CellFactory", lines: ["reads BaseConfig", "creates initial Spheroids"], x: 2790, yTop: 260, headerColor: color(255, 202, 212), bodyColor: color(255, 253, 248)),
    "Frame": Box(title: "Frame", lines: ["owns cells", "renders synthetic stack", "perturb()/trySplitCell()"], x: 2790, yTop: 860, headerColor: color(244, 172, 183), bodyColor: color(255, 253, 248)),
    "Lineage": Box(title: "Lineage", lines: ["owns frames", "optimizes and copies cells"], x: 2790, yTop: 1380, headerColor: color(244, 172, 183), bodyColor: color(255, 253, 248)),
    "LineageViewer": Box(title: "LineageViewer", lines: ["consumes frame snapshots", "visualizes lineage state"], x: 2790, yTop: 1900, headerColor: color(255, 229, 236), bodyColor: color(255, 253, 248)),
]

func drawText(_ text: String, in rect: NSRect, fontSize: CGFloat, weight: NSFont.Weight = .regular, color: NSColor = textColor, alignment: NSTextAlignment = .left) {
    let style = NSMutableParagraphStyle()
    style.alignment = alignment
    let attrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: fontSize, weight: weight),
        .foregroundColor: color,
        .paragraphStyle: style,
    ]
    text.draw(in: rect, withAttributes: attrs)
}

func fillRoundedRect(_ rect: NSRect, radius: CGFloat, color: NSColor) {
    let path = NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius)
    color.setFill()
    path.fill()
}

func strokeRoundedRect(_ rect: NSRect, radius: CGFloat, color: NSColor, width: CGFloat) {
    let path = NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius)
    path.lineWidth = width
    color.setStroke()
    path.stroke()
}

func drawBox(_ box: Box) {
    fillRoundedRect(box.rect, radius: 20, color: box.bodyColor)
    strokeRoundedRect(box.rect, radius: 20, color: border, width: 3)

    let headerRect = NSRect(x: box.rect.minX + 1.5, y: box.rect.maxY - headerHeight - 1.5, width: box.rect.width - 3, height: headerHeight)
    fillRoundedRect(headerRect, radius: 18, color: box.headerColor)

    let separator = NSBezierPath()
    separator.move(to: NSPoint(x: box.rect.minX, y: box.rect.maxY - headerHeight))
    separator.line(to: NSPoint(x: box.rect.maxX, y: box.rect.maxY - headerHeight))
    separator.lineWidth = 3
    border.setStroke()
    separator.stroke()

    drawText(box.title, in: NSRect(x: box.rect.minX, y: box.rect.maxY - 37, width: box.rect.width, height: 30), fontSize: 27, weight: .bold, color: titleColor, alignment: .center)

    for (index, line) in box.lines.enumerated() {
        drawText(line, in: NSRect(x: box.rect.minX + 16, y: box.rect.maxY - 74 - CGFloat(index) * 24, width: box.rect.width - 32, height: 22), fontSize: 20)
    }
}

func drawCluster(x: CGFloat, yTop: CGFloat, w: CGFloat, h: CGFloat, title: String, fill: NSColor) {
    let rect = NSRect(x: x, y: height - yTop - h, width: w, height: h)
    fillRoundedRect(rect, radius: 26, color: fill)
    strokeRoundedRect(rect, radius: 26, color: color(122, 139, 143), width: 2)
    drawText(title, in: NSRect(x: rect.minX + 24, y: rect.maxY - 42, width: 260, height: 30), fontSize: 34, weight: .bold, color: titleColor)
}

func line(_ from: NSPoint, _ to: NSPoint, dashed: Bool = false, width: CGFloat = 4) {
    let path = NSBezierPath()
    path.move(to: from)
    path.line(to: to)
    path.lineWidth = width
    if dashed {
        path.setLineDash([10, 8], count: 2, phase: 0)
    }
    arrowColor.setStroke()
    path.stroke()
}

func drawArrowHead(at tip: NSPoint, angle: CGFloat, filled: Bool) {
    let size: CGFloat = filled ? 20 : 24
    let spread: CGFloat = .pi / 7
    let left = NSPoint(x: tip.x - cos(angle - spread) * size, y: tip.y - sin(angle - spread) * size)
    let right = NSPoint(x: tip.x - cos(angle + spread) * size, y: tip.y - sin(angle + spread) * size)
    let path = NSBezierPath()
    path.move(to: tip)
    path.line(to: left)
    path.line(to: right)
    path.close()
    if filled {
        arrowColor.setFill()
        path.fill()
    } else {
        NSColor.white.setFill()
        path.fill()
        path.lineWidth = 4
        arrowColor.setStroke()
        path.stroke()
    }
}

func connect(_ from: NSPoint, _ to: NSPoint, label: String, dashed: Bool = false, inheritance: Bool = false, labelOffsetX: CGFloat = 0, labelOffsetY: CGFloat = 0) {
    line(from, to, dashed: dashed)
    let angle = atan2(to.y - from.y, to.x - from.x)
    drawArrowHead(at: to, angle: angle, filled: !inheritance)
    let mid = NSPoint(x: (from.x + to.x) / 2 + labelOffsetX, y: (from.y + to.y) / 2 + labelOffsetY)
    drawText(label, in: NSRect(x: mid.x - 90, y: mid.y - 12, width: 180, height: 24), fontSize: 22, weight: .semibold, color: color(71, 93, 99), alignment: .center)
}

func route(_ points: [NSPoint], dashed: Bool = false, width: CGFloat = 4, inheritance: Bool = false) {
    guard points.count >= 2 else { return }
    for i in 0..<(points.count - 1) {
        line(points[i], points[i + 1], dashed: dashed, width: width)
    }
    let a = points[points.count - 2]
    let b = points[points.count - 1]
    let angle = atan2(b.y - a.y, b.x - a.x)
    drawArrowHead(at: b, angle: angle, filled: !inheritance)
}

func labelAt(_ point: NSPoint, _ text: String) {
    drawText(text, in: NSRect(x: point.x - 95, y: point.y - 12, width: 190, height: 24), fontSize: 20, weight: .semibold, color: color(71, 93, 99), alignment: .center)
}

let image = NSImage(size: NSSize(width: width, height: height))
image.lockFocus()

fillRoundedRect(NSRect(x: 0, y: 0, width: width, height: height), radius: 0, color: color(248, 251, 255))
fillRoundedRect(NSRect(x: 0, y: 0, width: width, height: height), radius: 0, color: NSColor(calibratedWhite: 1.0, alpha: 0.88))

drawText("CellUniverse C++ Current Class Relationships", in: NSRect(x: 120, y: height - 120, width: 1200, height: 50), fontSize: 52, weight: .bold, color: titleColor)
drawText("Focused on actual class ownership, inheritance, and configuration dependencies in the current codebase.", in: NSRect(x: 120, y: height - 172, width: 1800, height: 30), fontSize: 28, color: color(79, 93, 117))

drawCluster(x: 50, yTop: 1220, w: 1520, h: 1160, title: "Cell Models", fill: color(255, 243, 224))
drawCluster(x: 1490, yTop: 140, w: 1110, h: 1280, title: "Configuration", fill: color(236, 254, 255))
drawCluster(x: 2660, yTop: 160, w: 770, h: 2020, title: "Core Pipeline", fill: color(255, 241, 242))

for key in ["Cell", "Bacilli", "Sphere", "Spheroid", "CellParams", "SphereParams", "SpheroidParams", "CellConfig", "SphereConfig", "SpheroidConfig", "PerturbParams", "SimulationConfig", "ProbabilityConfig", "BaseConfig", "CellFactory", "Frame", "Lineage", "LineageViewer"] {
    drawBox(boxes[key]!)
}

connect(NSPoint(x: boxes["Bacilli"]!.center.x, y: boxes["Bacilli"]!.rect.maxY), NSPoint(x: boxes["Cell"]!.center.x, y: boxes["Cell"]!.rect.minY), label: "inherits", inheritance: true, labelOffsetX: 50)
route([
    NSPoint(x: boxes["SphereParams"]!.rect.minX, y: boxes["SphereParams"]!.center.y),
    NSPoint(x: 1060, y: boxes["SphereParams"]!.center.y),
    NSPoint(x: 1060, y: boxes["CellParams"]!.center.y - 10),
    NSPoint(x: boxes["CellParams"]!.rect.maxX, y: boxes["CellParams"]!.center.y - 10)
], inheritance: true)
labelAt(NSPoint(x: 1030, y: 930), "inherits")

route([
    NSPoint(x: boxes["SpheroidParams"]!.rect.minX, y: boxes["SpheroidParams"]!.center.y),
    NSPoint(x: 1020, y: boxes["SpheroidParams"]!.center.y),
    NSPoint(x: 1020, y: boxes["CellParams"]!.center.y - 40),
    NSPoint(x: boxes["CellParams"]!.rect.maxX, y: boxes["CellParams"]!.center.y - 40)
], inheritance: true)
labelAt(NSPoint(x: 995, y: 650), "inherits")

route([
    NSPoint(x: boxes["SphereConfig"]!.center.x, y: boxes["SphereConfig"]!.rect.maxY),
    NSPoint(x: boxes["SphereConfig"]!.center.x, y: boxes["CellConfig"]!.center.y - 20),
    NSPoint(x: boxes["CellConfig"]!.center.x - 60, y: boxes["CellConfig"]!.center.y - 20),
    NSPoint(x: boxes["CellConfig"]!.center.x - 60, y: boxes["CellConfig"]!.rect.minY)
], inheritance: true)
labelAt(NSPoint(x: 1860, y: 2065), "inherits")

route([
    NSPoint(x: boxes["SpheroidConfig"]!.center.x + 20, y: boxes["SpheroidConfig"]!.rect.maxY),
    NSPoint(x: boxes["SpheroidConfig"]!.center.x + 20, y: boxes["CellConfig"]!.center.y + 10),
    NSPoint(x: boxes["CellConfig"]!.center.x + 65, y: boxes["CellConfig"]!.center.y + 10),
    NSPoint(x: boxes["CellConfig"]!.center.x + 65, y: boxes["CellConfig"]!.rect.minY)
], inheritance: true)
labelAt(NSPoint(x: 1940, y: 1910), "inherits")

connect(NSPoint(x: boxes["Sphere"]!.rect.maxX, y: boxes["Sphere"]!.center.y - 10), NSPoint(x: boxes["SphereParams"]!.rect.minX, y: boxes["SphereParams"]!.center.y), label: "uses", dashed: true, labelOffsetY: 24)
connect(NSPoint(x: boxes["Spheroid"]!.rect.maxX, y: boxes["Spheroid"]!.center.y + 10), NSPoint(x: boxes["SpheroidParams"]!.rect.minX, y: boxes["SpheroidParams"]!.center.y), label: "uses", dashed: true, labelOffsetY: -24)

route([
    NSPoint(x: boxes["SphereConfig"]!.center.x, y: boxes["SphereConfig"]!.rect.minY),
    NSPoint(x: boxes["SphereConfig"]!.center.x, y: boxes["PerturbParams"]!.center.y + 15),
    NSPoint(x: boxes["PerturbParams"]!.rect.maxX - 40, y: boxes["PerturbParams"]!.center.y + 15)
])
labelAt(NSPoint(x: 1800, y: 1470), "has many")

route([
    NSPoint(x: boxes["SpheroidConfig"]!.center.x + 15, y: boxes["SpheroidConfig"]!.rect.minY),
    NSPoint(x: boxes["SpheroidConfig"]!.center.x + 15, y: boxes["PerturbParams"]!.center.y - 15),
    NSPoint(x: boxes["PerturbParams"]!.rect.maxX - 80, y: boxes["PerturbParams"]!.center.y - 15)
])
labelAt(NSPoint(x: 1850, y: 1150), "has many")
connect(NSPoint(x: boxes["BaseConfig"]!.rect.minX, y: boxes["BaseConfig"]!.center.y + 34), NSPoint(x: boxes["SimulationConfig"]!.rect.maxX, y: boxes["SimulationConfig"]!.center.y), label: "owns")
connect(NSPoint(x: boxes["BaseConfig"]!.rect.minX, y: boxes["BaseConfig"]!.center.y - 34), NSPoint(x: boxes["ProbabilityConfig"]!.rect.maxX, y: boxes["ProbabilityConfig"]!.center.y), label: "owns")
connect(NSPoint(x: boxes["BaseConfig"]!.center.x - 50, y: boxes["BaseConfig"]!.rect.maxY), NSPoint(x: boxes["SpheroidConfig"]!.center.x + 55, y: boxes["SpheroidConfig"]!.rect.minY), label: "points to", labelOffsetX: 40, labelOffsetY: -16)
connect(NSPoint(x: boxes["CellFactory"]!.rect.minX, y: boxes["CellFactory"]!.center.y), NSPoint(x: boxes["BaseConfig"]!.rect.maxX, y: boxes["BaseConfig"]!.center.y + 8), label: "reads", dashed: true)

route([
    NSPoint(x: boxes["CellFactory"]!.rect.minX, y: boxes["CellFactory"]!.center.y - 20),
    NSPoint(x: 2580, y: boxes["CellFactory"]!.center.y - 20),
    NSPoint(x: 2580, y: boxes["Spheroid"]!.center.y + 30),
    NSPoint(x: boxes["Spheroid"]!.rect.maxX, y: boxes["Spheroid"]!.center.y + 30)
], dashed: true)
labelAt(NSPoint(x: 2580, y: 1280), "creates")

route([
    NSPoint(x: boxes["Frame"]!.rect.minX, y: boxes["Frame"]!.center.y + 20),
    NSPoint(x: 2660, y: boxes["Frame"]!.center.y + 20),
    NSPoint(x: 2660, y: boxes["SimulationConfig"]!.center.y + 10),
    NSPoint(x: boxes["SimulationConfig"]!.rect.maxX, y: boxes["SimulationConfig"]!.center.y + 10)
], dashed: true)
labelAt(NSPoint(x: 2660, y: 1740), "renders with")

route([
    NSPoint(x: boxes["Frame"]!.rect.minX, y: boxes["Frame"]!.center.y - 20),
    NSPoint(x: 2480, y: boxes["Frame"]!.center.y - 20),
    NSPoint(x: 2480, y: boxes["Spheroid"]!.center.y),
    NSPoint(x: boxes["Spheroid"]!.rect.maxX, y: boxes["Spheroid"]!.center.y)
])
labelAt(NSPoint(x: 2480, y: 1400), "owns cells")

route([
    NSPoint(x: boxes["Lineage"]!.rect.minX, y: boxes["Lineage"]!.center.y + 15),
    NSPoint(x: 2580, y: boxes["Lineage"]!.center.y + 15),
    NSPoint(x: 2580, y: boxes["BaseConfig"]!.center.y - 15),
    NSPoint(x: boxes["BaseConfig"]!.rect.maxX, y: boxes["BaseConfig"]!.center.y - 15)
])
labelAt(NSPoint(x: 2580, y: 1230), "stores copy")

connect(NSPoint(x: boxes["Lineage"]!.center.x, y: boxes["Lineage"]!.rect.maxY), NSPoint(x: boxes["Frame"]!.center.x, y: boxes["Frame"]!.rect.minY), label: "owns frames", labelOffsetX: 110)

route([
    NSPoint(x: boxes["Lineage"]!.rect.minX, y: boxes["Lineage"]!.center.y - 20),
    NSPoint(x: 2460, y: boxes["Lineage"]!.center.y - 20),
    NSPoint(x: 2460, y: boxes["Spheroid"]!.center.y - 30),
    NSPoint(x: boxes["Spheroid"]!.rect.maxX, y: boxes["Spheroid"]!.center.y - 30)
], dashed: true)
labelAt(NSPoint(x: 2460, y: 980), "seeds / forwards")

connect(NSPoint(x: boxes["LineageViewer"]!.center.x, y: boxes["LineageViewer"]!.rect.maxY), NSPoint(x: boxes["Lineage"]!.center.x, y: boxes["Lineage"]!.rect.minY), label: "visualizes", dashed: true, labelOffsetX: 110)

drawText("Note: in the current headers, Spheroid and Sphere do not inherit from Cell; the active tracking path uses Spheroid directly.", in: NSRect(x: 120, y: 80, width: 2400, height: 32), fontSize: 26, weight: .semibold, color: noteColor)

image.unlockFocus()

guard let tiff = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let png = bitmap.representation(using: .png, properties: [:]) else {
    fputs("Failed to encode PNG\n", stderr)
    exit(1)
}

let output = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    .appendingPathComponent("docs/current_class_relationships_a4.png")

try png.write(to: output)
print(output.path)
