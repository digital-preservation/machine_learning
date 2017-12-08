{
  :ssh                         => {
    :keys => ["id_rsa_acceptance", "#{ENV['HOME']}/.ssh/id_rsa-acceptance"],
  },
  :load_path                   => './lib',
  :xml                         => true,
  :timesync                    => false,
  :repo_proxy                  => true,
  :add_el_extras               => false,
  :forge_host                  => 'forge-aio01-petest.puppetlabs.com',
  :'master-start-curl-retries' => 30,
}
