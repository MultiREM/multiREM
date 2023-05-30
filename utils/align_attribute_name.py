distilled_attribute_names = {
  '16_s_recovered': ['16S recovered'], # 'yes', 'no'
  '16_s_recovery_software': ['16S recovery software'],
  'agent': ['agent'],
  'alkalinity_carbonate_bicarbonate': ['alkalinity (carbonate/bicarbonate)'],
  'alt_elev': ['alt_elev'],
  'altitude': ['altitude'],
  'aluminimum': ['Aluminimum'], # 366, 104, 132, ...integers
  'ammonium': ['ammonium'],
  'analysis_project_type': ['analysis project type'],
  'annotation_method': ['annotation_method'],
  'assembly_method_and_version': ['assembly_method_and_version'],
  'assembly_method_version': ['assembly_method_version'],
  'assembly_method': ['assembly method'],
  'assembly_method': ['assembly_method'],
  'assembly_quality': ['assembly quality'],
  # 'assembly_software': ['assembly software'],
  'assembly': ['assembly'],
  'attributes': ['attributes'],
  'bin_id': ['bin_id'],
  'bin_number': ['bin number'],
  'bin_parameters': ['bin parameters'],
  # 'binning_parameters': ['binning parameters'],
  # 'binning_software': ['binning software'],
  'bio_project_for_reads': ['BioProject for reads'], # This is different than the bioproject link attribute
  'bio_sample_for_sra_reads': ['BioSample for SRA reads'],
  'biomaterial_provider': ['biomaterial_provider'],
  'biome': ['biome'],
  'biosamplemodel': ['biosamplemodel'],
  'biotic_relationship': ['biotic_relationship'],
  'broad_scale_environmental_context': ['broad-scale environmental context'],
  'broker_name': ['broker name'],
  'calcium': ['Calcium'],
  'carbon_source': ['carbon_source'],
  'cell_shape': ['Cell Shape'],
  'chl_a_ugper_l': ['Chl-a_ugperL'],
  'chloride': ['chloride'],
  'chlorophyll': ['chlorophyll'],
  # 'collected_by': ['collected_by'],
  # 'collection_data': ['collection data'],
  # 'collection_date': ['collection date', 'collection_date', 'collection-date'],
  'completeness_estimated': ['completeness_estimated', 'completeness score'],
  # 'completeness_software': ['completeness software'],
  'condition': ['condition'],
  'conductivity': ['conductivity.', 'conductivity', 'conduc'],
  'contamination_estimated': ['contamination_estimated', 'contamination score'],
  'contig_l50': ['contig L50'],
  'copper_concentration': ['copper_concentration'],
  'copper': ['copper'],
  'country': ['country'],
  'culture_collection': ['culture_collection', 'culture-collection'],
  'culture_type': ['culture type'],
  'culture': ['culture'],
  'decontamination_software': ['decontamination software'],
  'depth_m': ['Depth (m)'], # integer
  'depth': ['depth'], # 1120m, 4-6cm, 4.5 meters
  'derived_from': ['derived from', 'derived_from', 'derived-from'],
  'description_gtdb_taxonomy': ['description_GTDB_taxonomy'],
  'description': ['description'],
  'development_stage': ['development stage'],
  'diseases': ['Diseases'], # only can be value disease lol
  'dissolved_carbon_dioxide': ['dissolved carbon dioxide'],
  'dissolved_organic_carbon': ['dissolved organic carbon'],
  'dna_treatment': ['dna treatment'],
  'do_mgper_l': ['DO_mgperL'],
  'e_lmsg_genome_id': ['eLMSG_genome_id'],
  'e_lmsg_link': ['eLMSG_link'],
  'elev': ['elev'],
  'elevation': ['elevation'],
  # 'ena_first_public': ['ENA first public', 'ENA-FIRST-PUBLIC'], # yyyy-mm-dd
  # 'ena_last_update': ['ENA last update', 'ENA-LAST-UPDATE'], # yyyy-mm-dd
  'env_biome': ['env_biome'],
  'env_broad_scale': ['env_broad_scale'],
  'env_feature': ['env_feature'],
  'env_local_scale': ['env_local_scale'],
  'env_material': ['env_material'],
  'env_medium': ['env_medium'],
  'env_package': ['env_package'],
  'environment_biome': ['environment (biome)'],
  'environment_feature': ['environment (feature)'],
  'environment_material': ['environment (material)'],
  'environmental_medium': ['environmental medium', 'environment'],
  # 'environmental_sample': ['environmental sample', 'environmental-sample', 'environmental_sample', 'environmetal-sample', 'envrionmental-sample', 'envirionmental-sample],
  'estimated_size': ['estimated_size'],
  'exp_condition_tag': ['exp_condition_tag'],
  'experimental_factor': ['experimental_factor'],
  'external_id': ['External Id'],
  'feature': ['feature'],
  'ferric_iron': ['ferric iron'],
  'ferrous_iron': ['ferrous iron'],
  'finishing_strategy': ['finishing_strategy'],
  'fluoride': ['fluoride'],
  'funding_program': ['Funding Program'],
  'gene_calling_method': ['Gene Calling Method'],
  'genome_coverage': ['genome_coverage'],
  'genotype': ['genotype', 'genotype/variation'], # LOOKS IMPORTANT: 'wild type'
  'geo_loc_name': ['geo_loc_name', 'geo-loc-name'],
  # these all seem very loose text
  'geographic_location_country_and_or_atlantic_ocean': ['geographic location (country and/or atlantic ocean)'],
  'geographic_location_country_and_or_sea': ['geographic location (country and/or sea)'],
  'geographic_location_depth': ['geographic location (depth)'],
  'geographic_location_elevation': ['geographic location (elevation)'],
  'geographic_location_latitude': ['geographic location (latitude)'],
  'geographic_location_longitude': ['geographic location (longitude)'],
  'geographic_location_region_and_locality': ['geographic location (region and locality)'],
  'geographic_location': ['geographic location'],
  'gold_stamp_id': ['GOLD Stamp ID'],
  'gram_staining': ['Gram Staining'],
  'greengenes_id': ['Greengenes ID'],
  'growth_conditions': ['growth conditions'], # LOOKS IMPORTANT: 'nitrate', 'ammonia, high oxygen', 'nitrate, low oxygen'
  'gtdb_taxonomy': ['gtdb_taxonomy'],
  'history': ['history'],
  'host_subject_id': ['host_subject_id'],
  'host_tissue_sampled': ['host_tissue_sampled'],
  'host': ['host'],
  'ice_thickness': ['ice thickness'],
  'id': ['ID'],
  # 'identified_by': ['identified_by'],
  # 'insdc_center_alias': ['INSDC center alias'],
  # 'insdc_center_name': ['INSDC center name'],
  # 'insdc_first_public': ['INSDC first public'],
  # 'insdc_last_update': ['INSDC last update'],
  # 'insdc_status': ['INSDC status'],
  'investigation_type': ['investigation type', 'investigation_type'],
  'iron': ['iron'],
  'isol_growth_condt': ['isol_growth_condt'],
  'isolate_name_alias': ['isolate_name_alias'],
  'isolate': ['isolate', 'Isolate'],
  'isolation_comments': ['Isolation Comments'],
  'isolation_site': ['Isolation Site'],
  'isolation_source': ['Isolation source', 'Isolation_source', 'isolation source', 'isolation_source', 'isolation-source'], # LOOKS IMPORTANT: 'sediment'
  'lake_code': ['Lake_code'],
  'lat_lon': ['lat_lon'], # duplicate of latitude, longitude
  'limitation': ['limitation'],
  'lineage': ['lineage'],
  'link': ['link'],
  'local_environmental_context': ['local environmental context'],
  'locus_tag_prefix': ['locus_tag prefix', 'locus_tag_prefix'],
  'mag': ['MAG'],
  'magnesium': ['magnesium'],
  'manganese': ['manganese'],
  'mapping_method_and_version': ['mapping_method_and_version'],
  'mapping_method_version': ['mapping_method_version'],
  'mapping_method': ['mapping_method'],
  'material': ['material'],
  'medium1_url': ['medium1_URL'],
  'medium1': ['medium1'],
  'medium2_url': ['medium2_URL'],
  'medium2': ['medium2'],
  'metagenome_source': ['metagenome_source', 'metagenome-source'],
  'metagenomic_otu': ['metagenomic OTU'],
  'metagenomic_source': ['metagenomic source', 'metagenomic-source'],
  'metagenomic': ['metagenomic'],
  'methane': ['methane'],
  'motility': ['Motility'],
  'mutation': ['mutation'],
  'name': ['name'],
  'nitrate': ['nitrate'],
  'nitrile_nitrite': ['nitrile+nitrite'],
  'nitrite': ['nitrite'],
  'nitrogen_source': ['nitrogen_source'],
  'nitrogen': ['nitrogen'],
  'note': ['note', 'notes'],
  'num_replicons': ['num_replicons'],
  'number_of_contigs': ['number of contigs'],
  'omics_observ_id': ['omics_observ_id'],
  'org_particles': ['org_particles'],
  'organic_carbon': ['organic carbon'],
  'orp_m_v': ['ORP_mV'],
  'other_cc': ['Other_CC'],
  'oxy_stat_samp': ['oxy_stat_samp'],
  'oxygen': ['oxygen'],
  'pacbio_sequencing_chemistry': ['pacbio sequencing chemistry'],
  'ph': ['ph', 'pH'],
  'phosphate': ['phosphate'],
  'potential_metabolism': ['Potential metabolism'],
  'project_name': ['project name', 'project_name'],
  'quality_assessment_method_and_version': ['quality_assessment_method_and_version'],
  'quality_assessment_method_version': ['quality_assessment_method_version'],
  'quality_assessment_method': ['quality_assessment_method'],
  'reactor': ['reactor'],
  'reassembly_post_binning': ['reassembly post binning'],
  'ref_biomaterial': ['ref_biomaterial'],
  'refinement_method': ['refinement_method'],
  'region': ['region'],
  'rel_to_oxygen': ['rel_to_oxygen'],
  'relative_coverage_anaerobic_srr10097246': ['relative coverage Anaerobic (SRR10097246)'],
  'relative_coverage_anoxic_srr10375098': ['relative coverage Anoxic (SRR10375098)'],
  'relative_coverage_infiltration_srr10097247': ['relative coverage Infiltration (SRR10097247)'],
  'relative_coverage_low_p_h_srr10097245': ['relative coverage Low pH (SRR10097245)'],
  'relative_coverage_oxic_srr10375099': ['relative coverage Oxic (SRR10375099)'],
  'relative_coverage_reference_srr10375097': ['relative coverage Reference (SRR10375097)'],
  'replicate': ['replicate'],
  'repository': ['repository'],
  'run_accession': ['run_accession'],
  'salinity_ppt': ['Salinity_ppt'],
  'salinity': ['salinity'],
  'samp_collect_device': ['samp_collect_device'],
  'samp_mat_process': ['samp_mat_process'],
  'samp_salinity': ['samp_salinity'],
  'samp_size': ['samp_size'],
  'samp_store_dur': ['samp_store_dur'],
  'samp_store_temp': ['samp_store_temp'],
  'sample_comment': ['sample comment'],
  'sample_derived_from': ['sample derived from'],
  'sample_name': ['sample name', 'sample_name'],
  'sample_number': ['Sample Number'],
  # 'sample_owner': ['sample owner'],
  'sample_type': ['sample type', 'sample-type', 'sample_type'],
  'scientific_name': ['scientific_name'],
  # 'sequencing_method': ['sequencing method', 'sequencing_meth'],
  'silicon': ['Silicon'],
  'site_name': ['site name'],
  'size_frac': ['size_frac'],
  'soil_environmental_package': ['soil environmental package'],
  'source_area': ['source_area'],
  'source_material_id': ['source_material_id'],
  'source_name': ['source_name'],
  'sporulation': ['Sporulation'],
  'sra_bio_samples': ['SRA_BioSamples'],
  'sra_experiment_accession': ['SRA experiment accession'],
  'sra_for_reads': ['SRA for reads'],
  'sra_run_accession': ['SRA run accession'],
  'sra': ['SRA'],
  'strain': ['strain', 'Strain', 'STRAIN'],
  'submitter_id': ['Submitter Id'],
  'subsource_note': ['subsource_note'],
  'subsrc_note': ['subsrc_note'],
  'sulfate': ['sulfate'],
  'suspended_solids_mgper_l': ['Suspended Solids_mgperL'],
  'swedish_lake_code': ['swedish lake code'],
  'tax_id': ['Tax ID'],
  'taxa_id': ['taxa id', 'taxonomic identity marker'],
  'temperature_cel': ['Temperature_CEL'],
  'temperature_from': ['temperature_from'],
  'temperature_optimum': ['Temperature Optimum'],
  'temperature_range': ['Temperature Range'],
  'temperature_to': ['temperature_to'],
  'temperature': ['temperature', 'temp'],
  'tissue': ['tissue'],
  'total_assembly_size': ['total assembly size'],
  'total_dissolved_nitrogen': ['total dissolved nitrogen'],
  'total_phosphorus': ['total phosphorus'],
  'treatment': ['treatment'],
  'trophic_level': ['trophic_level'],
  'turbidity_ntu': ['Turbidity_NTU'],
  'type_material': ['type-material'],
  'type_status': ['type_status'],
  'type_strain': ['Type Strain'],
  'unique_genome_name': ['unique_genome_name'],
  'update': ['update'],
  'validity': ['validity'],
  'value': ['value'],
  'wastewater_type': ['wastewater_type'],
  'water_environmental_package': ['water environmental package'],
}

def align_attribute_name(input_str):
    # loop through values on each key, and return snake case match (don't just snake case bc there were typos)
    for key, value in distilled_attribute_names.items():
        if input_str in value:
            return key
    return None
    # raise f"Unexpected input_str for attribute name: '{input_str}'"
