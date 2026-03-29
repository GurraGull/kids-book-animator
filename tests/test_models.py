from pipeline.models import PagePlan, BookPlan


def test_page_plan_no_character():
    page = PagePlan(filename="page01.jpg", character=False, action=None, description=None)
    assert page.character is False
    assert page.action is None


def test_page_plan_with_character():
    page = PagePlan(filename="page02.jpg", character=True, action="walk", description="girl on log")
    assert page.character is True
    assert page.action == "walk"
    assert page.description == "girl on log"


def test_book_plan_pages():
    pages = [
        PagePlan(filename="page01.jpg", character=False, action=None, description=None),
        PagePlan(filename="page02.jpg", character=True, action="walk", description="rabbit hopping"),
    ]
    book = BookPlan(title="Peter Rabbit", pages=pages)
    assert len(book.pages) == 2
    assert book.pages[1].action == "walk"
