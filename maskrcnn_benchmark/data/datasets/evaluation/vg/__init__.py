from .vg_eval import do_vg_evaluation


def vg_evaluation(
    cfg=None,
    dataset=None,
    predictions=None,
    output_folder=None,
    logger=None,
    iou_types=None,
    writer=None,
    iteration=None,
    experiment=None,
    **_
):
    if writer is None:
        raise ValueError('writer is None')
    if iteration is None:
        raise ValueError('iteration is None')

    return do_vg_evaluation(
        cfg=cfg,
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        iou_types=iou_types,
        writer=writer,
        iteration=iteration,
        experiment=experiment,
    )
