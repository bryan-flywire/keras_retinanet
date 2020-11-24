#!/usr/bin/env python3

import argparse
import pytator
import os
import pandas as pd
import progressbar

from pprint import pprint

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=
                                     "Sets up a training area from tator project")
    parser = pytator.tator.cli_parser(parser)
    parser.add_argument("--section", nargs='*', help='sections to download')
    parser.add_argument("--box-type-id", type=int, required=True)
    parser.add_argument("--keyname", default="Species", help="Key name to use for the species value in the box")
    parser.add_argument("--squash-species", action="store_true",
                        help="Squash species annotations into a single class")
    parser.add_argument("--squash-species-name", default="object", help="name to use for the single class")

    parser.add_argument("output_dir")
    args = parser.parse_args()

    tator = pytator.Tator(args.url, args.token, args.project)

    localizations = tator.Localization.filter({"type": args.box_type_id})
    os.makedirs(args.output_dir, exist_ok=True)
    species_output=os.path.join(args.output_dir, "species.csv")
    if args.squash_species:
        single_species=[(args.squash_species_name, 0)]
        species_df = pd.DataFrame(data=single_species,
                                  columns=['Species','Num'])
        species_df.to_csv(species_output, index=False, header=False)
    else:
        # First get the species for the project by getting the set of the
        # species names
        species = {}
        for localization in localizations:
            species_name = localization['attributes'][args.keyname]
            if species_name in species:
                species[species_name] += 1
            else:
                species[species_name] = 1

        print("Species Information:")
        pprint(species)

        names=list(species.keys())
        names.sort()
        name_idx_pairs=[]
        for idx,name in enumerate(names):
            name_idx_pairs.append((name,idx))
        species_df = pd.DataFrame(data=name_idx_pairs,
                                  columns=['Species','Num'])
        species_df.to_csv(species_output, index=False, header=False)


    # generate annotations.csv
    annotations_output=os.path.join(args.output_dir, "totalPopulation.csv")
    # output is img, x1,y1,x2,y2,species-id-0
    images_dir=os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    cols=['img', 'x1', 'y1', 'x2', 'y2', 'species_name']
    local_df=pd.DataFrame(columns=cols)
    local_df.to_csv(annotations_output, header=False,index=False)

    if args.section:
        sections_to_process=args.section
    else:
        sections_to_process=list(tator.MediaSection.all().keys())

    print(f"Processing {sections_to_process}")

    for section in sections_to_process:
        section_filter=f"tator_user_sections::{section}"
        medias = tator.Media.filter({'attribute': section_filter})
        section_dir=os.path.join(images_dir, section)
        os.makedirs(section_dir, exist_ok=True)
        bar = progressbar.ProgressBar(redirect_stdout=True)
        for media in bar(medias):
            media_element = tator.Media.get(media['id'])
            localizations= tator.Localization.filter({"type": args.box_type_id,
                                                      "media_id": media['id']})

            if localizations is None:
                print(f"{media_element['name']}({media_element['id']}) has no localizations")
                continue
            for localization in localizations:
                frame = localization['frame']
                image_path = os.path.join(section_dir, f"{media_element['name']}_{frame}.png")
                rel_image_path = os.path.relpath(image_path, images_dir)
                if not os.path.exists(image_path):
                    # TODO Determine if image or video
                    _,image_data = tator.GetFrame.get_encoded_img(media_element, [frame])
                    print(f"Saving {image_path}")
                    with open(image_path,'wb') as image_fp:
                        image_fp.write(image_data)
                shape = media_element['media_files']['streaming'][0]['resolution']
                img_width = shape[1]
                img_height = shape[0]
                x1 = localization['x'] * img_width
                y1 = localization['y'] * img_height
                width = localization['width'] * img_width
                height = localization['height'] * img_height
                x2 = x1 + width
                y2 = y1 + height
                if args.squash_species:
                    species_name = args.squash_species_name
                else:
                    species_name = localization['attributes'][args.keyname]

                x1 = max(x1,0)
                x2 = min(x2, img_width)
                y1 = max(y1,0)
                y2 = min(y2, img_height)
                datum={'img': rel_image_path,
                       'x1': round(x1),
                       'y1': round(y1),
                       'x2': round(x2),
                       'y2': round(y2),
                       'species_name': species_name}
                datum_df = pd.DataFrame(data=[datum],
                                 columns=cols)
                datum_df.to_csv(annotations_output,
                                mode='a',
                                header=False,
                                index=False)
