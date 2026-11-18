from app.cli import main

def test_hello(capsys):
    main(["hello"])
    assert "ready" in capsys.readouterr().out

def test_prepare_and_mine_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["prepare", "--per-role", "3"])
    main(["mine"])
    output = capsys.readouterr().out
    assert "Frequent itemsets" in output
    assert (tmp_path / "data/mining.json").exists()
