// Local-only OCR. Coordinates in output are normalized, top-left origin.
import Foundation
import Vision
import AppKit
import PDFKit
import ImageIO

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(message.utf8))
    exit(1)
}

let arguments = CommandLine.arguments
guard arguments.count >= 4 else { fail("usage: vision-ocr path page languages") }
let url = URL(fileURLWithPath: arguments[1])
let pageNumber = Int(arguments[2]) ?? 0
var image: CGImage
var imageCount = 1
if pageNumber > 0 {
    guard let document = PDFDocument(url: url),
          let page = document.page(at: pageNumber - 1) else { fail("Cannot open PDF page") }
    let bounds = page.bounds(for: .cropBox)
    guard bounds.width > 0 && bounds.height > 0 else { fail("Invalid PDF page bounds") }
    let scale = min(200.0 / 72.0, 4096.0 / max(bounds.width, bounds.height))
    let rendered = page.thumbnail(of: NSSize(width: bounds.width * scale,
                                             height: bounds.height * scale), for: .cropBox)
    var rect = CGRect(origin: .zero, size: rendered.size)
    guard let cg = rendered.cgImage(forProposedRect: &rect, context: nil, hints: nil)
        else { fail("Cannot render PDF page") }
    image = cg
} else {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { fail("Cannot open image") }
    imageCount = CGImageSourceGetCount(source)
    guard imageCount == 1 else { fail("Multi-frame images must be converted to PDF; refusing first-frame-only OCR") }
    let options: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: 4096
    ]
    guard let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
        else { fail("Cannot decode image") }
    image = cg
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
// Avoid rewriting identifiers/parameters through dictionary corrections.
request.usesLanguageCorrection = false
let wanted = arguments[3].split(separator: ",").map(String.init)
do {
    let supported = try request.supportedRecognitionLanguages()
    guard wanted.allSatisfy({ supported.contains($0) }) else { fail("Unsupported OCR language") }
    request.recognitionLanguages = wanted
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    let lines: [[String: Any]] = (request.results ?? []).compactMap { observation in
        guard let candidate = observation.topCandidates(1).first else { return nil }
        let box = observation.boundingBox
        return ["text": candidate.string, "confidence": candidate.confidence,
                "bbox": [Double(box.minX), Double(1.0 - box.maxY), Double(box.width), Double(box.height)]]
    }.sorted {
        let a = $0["bbox"] as! [Double], b = $1["bbox"] as! [Double]
        if abs(a[1] - b[1]) > 0.01 { return a[1] < b[1] }
        return a[0] < b[0]
    }
    let result: [String: Any] = ["engine": "apple-vision", "image_count": imageCount, "width": image.width,
                                "height": image.height, "lines": lines]
    let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
} catch { fail("Vision OCR failed: \(error)") }
