import { onCleanup, onMount } from "solid-js";

interface Point {
  x: number;
  y: number;
}

type Line = Point[];

function Canvas() {
  let canvas: HTMLCanvasElement | undefined;
  let requestID: number | undefined;

  let center: Point | undefined;

  let drawing = false;
  let lines: Line[] = [];
  let currentLine: Line = [];

  const startDrawing = (e: MouseEvent) => {
    if (!canvas || !center) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * devicePixelRatio - center.x;
    const y = (e.clientY - rect.top) * devicePixelRatio - center.y;
    drawing = true;
    currentLine = [{ x, y }];
    lines.push(currentLine);
  };

  const drawLine = (e: MouseEvent) => {
    if (!drawing || !canvas || !center) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * devicePixelRatio - center.x;
    const y = (e.clientY - rect.top) * devicePixelRatio - center.y;
    currentLine.push({ x, y });
  };

  const endDrawing = () => {
    drawing = false;
  };

  const resetDrawing = (e: KeyboardEvent) => {
    lines = [];
  };

  const exportLinesToJson = () => {
    const converted = lines.map((line) =>
      line.map((point) => [point.x, point.y])
    );
    const json = JSON.stringify(converted);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const filename = prompt("filename to save", "lines.json");
    if (!filename) return;

    const a = document.createElement("a");
    a.setAttribute("href", url);
    a.setAttribute("download", filename);
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();

    URL.revokeObjectURL(url);
  };

  const onKeydown = (e: KeyboardEvent) => {
    switch (e.key) {
      case "r":
        resetDrawing(e);
        break;
      case "s":
        exportLinesToJson();
        break;
      default:
        break;
    }
  };

  onMount(() => {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", drawLine);
    canvas.addEventListener("mouseup", endDrawing);

    window.addEventListener("keydown", onKeydown);

    const frame = () => {
      if (!canvas || !ctx) return;
      const currentWidth = canvas.clientWidth * devicePixelRatio;
      const currentHeight = canvas.clientHeight * devicePixelRatio;

      if (
        (currentWidth !== canvas.width || currentHeight !== canvas.height) &&
        currentWidth &&
        currentHeight
      ) {
        canvas.width = currentWidth;
        canvas.height = currentHeight;
      }

      center = prepare(ctx);

      ctx.strokeStyle = "black";
      ctx.lineWidth = 3;
      lines.forEach((line, i) => {
        if (i === lines.length - 1) {
          ctx.strokeStyle = "green";
        } else {
          ctx.strokeStyle = "rgba(0, 0, 0, 0.4)";
        }
        ctx.beginPath();
        line.forEach((point) => {
          if (!center) return;
          ctx.lineTo(point.x + center.x, point.y + center.y);
        });
        ctx.stroke();
      });

      requestID = requestAnimationFrame(frame);
    };

    requestID = requestAnimationFrame(frame);
  });

  onCleanup(() => {
    if (canvas) {
      canvas.removeEventListener("mousedown", startDrawing);
      canvas.removeEventListener("mousemove", drawLine);
      canvas.removeEventListener("mouseup", endDrawing);
    }
    requestID && cancelAnimationFrame(requestID);
  });

  return (
    <div class="flex-1">
      <canvas ref={canvas} class="h-full w-full" />
    </div>
  );
}

const prepare = (ctx: CanvasRenderingContext2D) => {
  const { width, height } = ctx.canvas;

  ctx.fillStyle = "gray";
  ctx.fillRect(0, 0, width, height);

  const whiteAreaWidth = width * 0.8;
  const whiteAreaHeight = height * 0.8;
  const whiteAreaX = (width - whiteAreaWidth) / 2;
  const whiteAreaY = (height - whiteAreaHeight) / 2;
  ctx.fillStyle = "white";
  ctx.fillRect(whiteAreaX, whiteAreaY, whiteAreaWidth, whiteAreaHeight);

  ctx.strokeStyle = "black";
  ctx.lineWidth = 1;
  const gridSpacing = 100;

  const center = {
    x: whiteAreaX + whiteAreaWidth / 2,
    y: whiteAreaY + whiteAreaHeight / 2,
  };

  ctx.beginPath();
  for (
    let offset = 0;
    center.x + offset < whiteAreaX + whiteAreaWidth;
    offset += gridSpacing
  ) {
    ctx.moveTo(center.x + offset, whiteAreaY);
    ctx.lineTo(center.x + offset, whiteAreaY + whiteAreaHeight);
    ctx.moveTo(center.x - offset, whiteAreaY);
    ctx.lineTo(center.x - offset, whiteAreaY + whiteAreaHeight);
  }
  for (
    let offset = 0;
    center.y + offset < whiteAreaY + whiteAreaHeight;
    offset += gridSpacing
  ) {
    ctx.moveTo(whiteAreaX, center.y + offset);
    ctx.lineTo(whiteAreaX + whiteAreaWidth, center.y + offset);
    ctx.moveTo(whiteAreaX, center.y - offset);
    ctx.lineTo(whiteAreaX + whiteAreaWidth, center.y - offset);
  }
  ctx.stroke();

  ctx.fillStyle = "red";
  ctx.beginPath();
  ctx.arc(center.x, center.y, 5, 0, Math.PI * 2);
  ctx.fill();

  return center;
};

export default Canvas;
