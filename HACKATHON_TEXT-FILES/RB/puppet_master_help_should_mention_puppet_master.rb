test_name "puppet master help should mention puppet master"

tag 'audit:medium',
    'audit:unit'

on master, puppet_master('--help') do
    fail_test "puppet master wasn't mentioned" unless stdout.include? 'puppet master'
end
