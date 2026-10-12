"""Gradio UI mounted at /ui. Submits into the same queue as the HTTP API."""
import os
import time

import gradio as gr

from .generate_ui import build_tabs
from .models import REPOS

COMPRESSIONS = ["piz", "zip16", "zip", "none"]
COLUMNS = ["id", "status", "progress", "model", "input", "submitted"]


CSS = """
.model-list { max-height: 300px; overflow-y: auto; }
.model-list .wrap { flex-direction: column; gap: 2px; }
.card { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px 14px; }
.status-line { font-size: 0.9em; opacity: 0.85; }
.hits { align-self: center; font-size: 0.85em; opacity: 0.75; }
.group-title { padding: 6px 10px 0; }
"""


# Neutral greys only; colour is left for status and the one primary action per step.
THEME = gr.themes.Base(primary_hue="zinc", secondary_hue="zinc", neutral_hue="zinc",
                       radius_size="sm",
                       font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "Ubuntu", "sans-serif"],
                       font_mono=["ui-monospace", "DejaVu Sans Mono", "monospace"]).set(
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="*neutral_500",
    block_label_text_color_dark="*neutral_400",
    block_label_border_width="0px",
    block_title_text_color="*neutral_600",
    block_title_background_fill="transparent",
    block_title_text_weight="500",
    block_border_width="1px",
    block_shadow="none",
    button_primary_background_fill="*neutral_800",
    button_primary_background_fill_hover="*neutral_700",
    button_primary_background_fill_dark="*neutral_200",
    button_primary_background_fill_hover_dark="*neutral_300",
    button_primary_text_color="white",
    button_primary_text_color_dark="*neutral_900",
    button_secondary_background_fill="white",
    button_secondary_background_fill_dark="*neutral_800",
    button_primary_shadow="none",
    button_secondary_shadow="none",
    button_secondary_border_color="*neutral_300",
    button_secondary_border_color_dark="*neutral_600",
    button_border_width="1px",
    checkbox_background_color_selected="*neutral_700",
    checkbox_border_color_selected="*neutral_700",
    checkbox_label_background_fill_selected="*neutral_100",
    checkbox_label_background_fill_selected_dark="*neutral_700",
    checkbox_label_text_color_selected="*neutral_900",
    checkbox_label_text_color_selected_dark="white",
    link_text_color="*neutral_700",
    link_text_color_dark="*neutral_300",
)


def build_ui():
    from . import server as svc

    def options(encoder, process_res, batch_size, compression, preview, first, last):
        return svc.JobOptions(
            encoder=encoder, process_res=int(process_res), batch_size=int(batch_size),
            compression=compression, preview=preview,
            first=None if first is None else int(first),
            last=None if last is None else int(last),
        )

    def submit_upload(file, *opts):
        if not file:
            raise gr.Error("Choose a video, image, or .zip of a sequence.")
        try:
            with open(file, "rb") as f:
                job = svc.submit_upload_job(f, os.path.basename(file), options(*opts))
        except ValueError as e:  # JobError and pydantic validation
            raise gr.Error(str(e))
        return job.id

    def submit_path(path, output_dir, *opts):
        if not path:
            raise gr.Error("Enter an input path.")
        try:
            req = svc.PathJob(input=path.strip(), output_dir=output_dir.strip() or None,
                              **options(*opts).model_dump())
            job = svc.submit_path_job(req)
        except ValueError as e:
            raise gr.Error(str(e))
        return job.id

    def table():
        rows = []
        for j in svc.jobs.list():
            progress = f"{j.done}/{j.total}" if j.total else ""
            if j.status == "queued":
                progress = f"#{svc.jobs.position(j)} in queue"
            rows.append([j.id, j.status, progress, j.encoder, j.input,
                         time.strftime("%H:%M:%S", time.localtime(j.created))])
        return rows

    def details(job_id):
        job = svc.jobs.get(job_id or "")
        if job is None:
            return "Submit a job or pick one from the list."
        lines = [f"**{job.id}** · `{job.encoder}` · **{job.status}**"]
        if job.total:
            lines.append(f"frames: {job.done}/{job.total}")
        if job.started:
            lines.append(f"time: {((job.finished or time.time()) - job.started):.1f}s")
        if job.error:
            lines.append(f"error: `{job.error}`")
        if job.result:
            lines.append(f"output: `{job.result['exr_pattern']}` "
                         f"({job.result['first']}-{job.result['last']})")
        return "  \n".join(lines)

    def refresh(job_id, shown):
        job = svc.jobs.get(job_id or "")
        key = f"{job_id}:{job.status if job else ''}"
        if key == shown:
            # don't reset the video player every tick
            return table(), details(job_id), gr.skip(), gr.skip(), shown
        video = download = None
        if job is not None and job.status == "done":
            video = job.result.get("preview")
            download = svc.result_zip(job)
        return table(), details(job_id), video, download, key

    def cancel(job_id):
        job = svc.jobs.get(job_id or "")
        if job is not None:
            svc.jobs.cancel(job)

    def pick(evt: gr.SelectData):
        return evt.row_value[0]

    # Gradio copies served files into its cache; drop copies older than a day, hourly.
    with gr.Blocks(title="Depth Anything V3", delete_cache=(3600, 86400)) as ui:
        gr.Markdown("# Depth Anything V3")
        with gr.Tabs() as tabs:
            with gr.Tab("Depth", id="depth"):
                gr.Markdown("Half-float EXR depth (`depth.Z`) + preview. "
                            "One job runs at a time; the rest wait in the queue.")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Tab("Upload"):
                            file = gr.File(label="Video, image, or .zip of an image sequence",
                                           type="filepath")
                            upload_btn = gr.Button("Submit upload", variant="primary")
                        with gr.Tab("Path on server"):
                            path = gr.Textbox(label="Input",
                                              placeholder="/projects/shot010/plate/shot010.####.exr")
                            output_dir = gr.Textbox(label="Output dir",
                                                    placeholder="default: <input dir>/da3_depth")
                            path_btn = gr.Button("Submit path", variant="primary")
                        with gr.Accordion("Options", open=True):
                            encoder = gr.Dropdown(list(REPOS), value="da3-base", label="Model")
                            process_res = gr.Slider(252, 2016, value=504, step=14,
                                                    label="Processing resolution")
                            batch_size = gr.Slider(1, 32, value=1, step=1,
                                                   label="Batch size (frames per inference)")
                            compression = gr.Dropdown(COMPRESSIONS, value="piz", label="EXR compression")
                            preview = gr.Checkbox(value=True, label="Write preview mp4")
                            with gr.Row():
                                first = gr.Number(label="First frame", precision=0)
                                last = gr.Number(label="Last frame", precision=0)
                    with gr.Column(scale=2):
                        with gr.Row():
                            job_id = gr.Textbox(label="Job", scale=3)
                            cancel_btn = gr.Button("Cancel job", variant="stop", scale=1)
                        info = gr.Markdown()
                        video = gr.Video(label="Preview", interactive=False)
                        download = gr.File(label="Result (EXRs + preview)", interactive=False)
                        jobs_table = gr.Dataframe(headers=COLUMNS, interactive=False, wrap=True,
                                                  label="Jobs (click a row)")
            gen = build_tabs(svc)
        shown = gr.State("")

        opts = [encoder, process_res, batch_size, compression, preview, first, last]
        upload_btn.click(submit_upload, [file, *opts], job_id)
        path_btn.click(submit_path, [path, output_dir, *opts], job_id)
        cancel_btn.click(cancel, job_id)
        jobs_table.select(pick, None, job_id)

        outputs = [jobs_table, info, video, download, shown]
        gr.Timer(2).tick(refresh, [job_id, shown], outputs, show_progress="hidden")
        job_id.change(refresh, [job_id, shown], outputs, show_progress="hidden")
        ui.load(refresh, [job_id, shown], outputs)
        gen["dup_btn"].click(gen["duplicate"], gen["sel"], [*gen["dup_outputs"], tabs])
        ui.load(gen["funds"], None, gen["funds_outputs"])
        ui.load(gen["load_form"], gen["workflow"], gen["form_outputs"])
    return ui
